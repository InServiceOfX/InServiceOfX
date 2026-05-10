import gc
from typing import Any, Dict, List, Optional, Sequence

from PIL import Image

from moremineru.Configurations import Qwen3VLConfiguration


class Qwen3VLVLLM:
    """Thin wrapper around vllm.LLM serving Qwen3-VL.

    Lifecycle mirrors MinerU2_5ProVLLM: construct cheaply, call load() before
    inference, call release() to free GPU. The vLLM engine handles the chat
    template (loaded from chat_template.json in the model dir) and the
    multi-modal image plumbing automatically when we use llm.chat(...).
    """

    def __init__(self, configuration: Qwen3VLConfiguration):
        self._configuration = configuration
        self._llm: Optional[Any] = None
        self._loaded = False

    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        if self._loaded:
            return

        from vllm import LLM

        engine_kwargs = dict(self._configuration.vllm_engine_kwargs)
        self._llm = LLM(
            model=str(self._configuration.model_path),
            **engine_kwargs,
        )
        self._loaded = True

    def _ensure_loaded(self, method: str) -> None:
        if not self._loaded:
            raise RuntimeError(
                f"Qwen3VLVLLM.{method} called before load()."
            )

    def _build_sampling_params(
        self,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> Any:
        from vllm import SamplingParams

        merged: Dict[str, Any] = dict(
            self._configuration.default_sampling_params
        )
        if overrides:
            merged.update(overrides)
        return SamplingParams(**merged)

    def _build_messages(
        self,
        image: Image.Image,
        prompt: str,
    ) -> List[Dict[str, Any]]:
        # ChatML format that vLLM 0.11.x feeds to the model's chat_template.
        # The chat template reads `image` entries from the user-role content
        # list and routes them through the multimodal preprocessor.
        messages: List[Dict[str, Any]] = []
        if self._configuration.system_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": self._configuration.system_prompt,
                }
            )
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        )
        return messages

    def generate(
        self,
        image: Image.Image,
        prompt: str,
        sampling_overrides: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Single image+prompt -> generated text. Greedy by default."""
        self._ensure_loaded("generate")
        messages = self._build_messages(image, prompt)
        sampling_params = self._build_sampling_params(sampling_overrides)
        outputs = self._llm.chat(
            [messages],
            sampling_params=sampling_params,
            use_tqdm=False,
        )
        # vllm.chat returns one RequestOutput per input; each holds a list of
        # CompletionOutput. We submitted a single conversation, take its first
        # candidate.
        return outputs[0].outputs[0].text

    def generate_batch(
        self,
        items: Sequence[Dict[str, Any]],
        sampling_overrides: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """Batched variant: each item is {"image": PIL.Image, "prompt": str}.

        vLLM 0.11.x batches multimodal inputs natively as long as we pass a
        list of conversations to llm.chat, so this is a real throughput win
        over a Python loop around generate(). Order of returns matches the
        order of items.
        """
        self._ensure_loaded("generate_batch")
        conversations = [
            self._build_messages(item["image"], item["prompt"])
            for item in items
        ]
        sampling_params = self._build_sampling_params(sampling_overrides)
        outputs = self._llm.chat(
            conversations,
            sampling_params=sampling_params,
            use_tqdm=False,
        )
        return [output.outputs[0].text for output in outputs]

    def release(self) -> None:
        self._llm = None
        self._loaded = False
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
