import gc
from typing import Any, List, Optional

from PIL import Image

from moremineru.Configurations import MinerUConfiguration


class MinerU2_5ProVLLM:
    def __init__(self, configuration: MinerUConfiguration):
        self._configuration = configuration
        self._llm: Optional[Any] = None
        self._client: Optional[Any] = None
        self._loaded = False

    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        if self._loaded:
            return

        # Imports are lazy so that constructing the wrapper (e.g. for a dry
        # run) doesn't require vLLM/CUDA to be importable.
        from vllm import LLM
        from mineru_vl_utils import MinerUClient, MinerULogitsProcessor

        engine_kwargs = dict(self._configuration.vllm_engine_kwargs)

        self._llm = LLM(
            model=str(self._configuration.model_path),
            logits_processors=[MinerULogitsProcessor],
            **engine_kwargs,
        )

        self._client = MinerUClient(
            backend=self._configuration.backend,
            vllm_llm=self._llm,
            image_analysis=self._configuration.image_analysis,
        )

        self._loaded = True

    def extract_from_image(self, image: Image.Image) -> Any:
        if not self._loaded:
            raise RuntimeError(
                "MinerU2_5ProVLLM.extract_from_image called before load()."
            )
        return self._client.two_step_extract(image)

    def extract_from_images(
        self,
        images: List[Image.Image],
    ) -> List[Any]:
        if not self._loaded:
            raise RuntimeError(
                "MinerU2_5ProVLLM.extract_from_images called before load()."
            )
        # MinerUClient.two_step_extract handles a single image at a time;
        # the vllm-async-engine path supports concurrent submission, but the
        # in-process LLM API does not. Iterate sequentially here.
        return [self._client.two_step_extract(image) for image in images]

    def release(self) -> None:
        self._client = None
        self._llm = None
        self._loaded = False
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
