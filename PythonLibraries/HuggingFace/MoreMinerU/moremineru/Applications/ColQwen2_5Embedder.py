"""ColQwen2.5 multi-vector retrieval embedder.

Wraps ``colpali_engine.models.ColQwen2_5`` (the LoRA-on-Qwen2.5-VL model)
plus ``ColQwen2_5_Processor``. Produces multi-vector embeddings (one vector
per image patch / text token) used with MaxSim scoring. Not a generation
model — its output is a tensor, not text.

Reference snippet (from https://huggingface.co/vidore/colqwen2.5-v0.2):

    from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
    model = ColQwen2_5.from_pretrained(model_path, ...).eval()
    processor = ColQwen2_5_Processor.from_pretrained(model_path)
    batch_images = processor.process_images(images).to(model.device)
    image_embeddings = model(**batch_images)
    batch_queries = processor.process_queries(queries).to(model.device)
    query_embeddings = model(**batch_queries)
    scores = processor.score_multi_vector(query_embeddings, image_embeddings)
"""
from __future__ import annotations

import gc
from typing import Any, List, Optional, Sequence

from PIL import Image

from moremineru.Configurations import ColQwen2_5Configuration


# Mapping from YAML-friendly torch dtype strings to ``torch.dtype`` values.
# Module-level so we don't reimport on every load().
_TORCH_DTYPE_BY_NAME = {
    "bfloat16": "bfloat16",
    "float16": "float16",
    "float32": "float32",
    "auto": "auto",
}


class ColQwen2_5Embedder:
    """Lifecycle mirrors MinerU2_5ProVLLM / Qwen3VLVLLM.

    Construct cheaply, ``load()`` before use, ``release()`` to free GPU.
    """

    def __init__(self, configuration: ColQwen2_5Configuration):
        self._configuration = configuration
        self._model: Optional[Any] = None
        self._processor: Optional[Any] = None
        self._loaded = False

    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        if self._loaded:
            return

        # Lazy imports: constructing the wrapper doesn't drag in torch/CUDA.
        import torch
        from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor

        dtype_str = _TORCH_DTYPE_BY_NAME.get(
            self._configuration.torch_dtype, self._configuration.torch_dtype
        )
        if dtype_str == "auto":
            torch_dtype = "auto"
        else:
            torch_dtype = getattr(torch, dtype_str)

        load_kwargs: dict = {
            "torch_dtype": torch_dtype,
            "device_map": self._configuration.device_map,
        }
        if self._configuration.attn_implementation is not None:
            load_kwargs["attn_implementation"] = (
                self._configuration.attn_implementation
            )

        self._model = ColQwen2_5.from_pretrained(
            str(self._configuration.model_path),
            **load_kwargs,
        ).eval()
        self._processor = ColQwen2_5_Processor.from_pretrained(
            str(self._configuration.model_path),
        )
        self._loaded = True

    def _ensure_loaded(self, method: str) -> None:
        if not self._loaded:
            raise RuntimeError(
                f"ColQwen2_5Embedder.{method} called before load()."
            )

    def embed_images(self, images: Sequence[Image.Image]) -> Any:
        """Batched image embedding. Returns a multi-vector tensor."""
        self._ensure_loaded("embed_images")
        import torch

        batch = self._processor.process_images(list(images)).to(
            self._model.device
        )
        with torch.no_grad():
            embeddings = self._model(**batch)
        return embeddings

    def embed_image(self, image: Image.Image) -> Any:
        """Single-image convenience. Returns one row of the batch tensor."""
        embeddings = self.embed_images([image])
        return embeddings[0]

    def embed_queries(self, texts: Sequence[str]) -> Any:
        """Batched text-query embedding. Returns a multi-vector tensor."""
        self._ensure_loaded("embed_queries")
        import torch

        batch = self._processor.process_queries(list(texts)).to(
            self._model.device
        )
        with torch.no_grad():
            embeddings = self._model(**batch)
        return embeddings

    def embed_query(self, text: str) -> Any:
        """Single-query convenience."""
        embeddings = self.embed_queries([text])
        return embeddings[0]

    def score(
        self,
        query_embeddings: Any,
        image_embeddings: Any,
    ) -> Any:
        """MaxSim scoring. Returns a ``[num_queries, num_images]`` tensor."""
        self._ensure_loaded("score")
        return self._processor.score_multi_vector(
            query_embeddings, image_embeddings
        )

    def release(self) -> None:
        self._model = None
        self._processor = None
        self._loaded = False
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
