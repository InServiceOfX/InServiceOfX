"""vLLM-served Qwen3-VL wrapper.

Follows the canonical inference pattern from the QwenLM/Qwen3-VL upstream
README: load ``transformers.AutoProcessor`` alongside ``vllm.LLM``, build the
prompt text via ``processor.apply_chat_template``, extract image inputs via
``qwen_vl_utils.process_vision_info``, then feed both to ``llm.generate``.

This is the path the Qwen team recommends; ``llm.chat`` (the higher-level vLLM
convenience) skips ``process_vision_info`` and can mis-handle Qwen3-VL's
specific image patching / resizing budget.
"""
from __future__ import annotations

import gc
from typing import Any, Dict, List, Optional, Sequence

from PIL import Image

from moremineru.Configurations import Qwen3VLConfiguration


class Qwen3VLVLLM:
    """Thin wrapper around vllm.LLM serving Qwen3-VL (bf16 or AWQ/INT4/INT8/FP8).

    Lifecycle mirrors ``MinerU2_5ProVLLM``: construct cheaply, ``load()`` before
    inference, ``release()`` to free GPU.
    """

    def __init__(self, configuration: Qwen3VLConfiguration):
        self._configuration = configuration
        self._llm: Optional[Any] = None
        self._processor: Optional[Any] = None
        self._loaded = False

    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        if self._loaded:
            return

        # Lazy imports so constructing the wrapper doesn't drag in vLLM/CUDA.
        from vllm import LLM
        from transformers import AutoProcessor

        model_path_str = str(self._configuration.model_path)
        engine_kwargs = dict(self._configuration.vllm_engine_kwargs)
        self._llm = LLM(model=model_path_str, **engine_kwargs)
        # The processor lives on CPU and is cheap; load from the same dir so
        # tokenizer + chat template + image processor versions all match.
        self._processor = AutoProcessor.from_pretrained(model_path_str)
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
        # Qwen3-VL expects ChatML user/system roles; vision inputs go under
        # the user content as ``{"type": "image", "image": <PIL.Image>}``.
        # Optional `max_pixels` / `min_pixels` are read by
        # ``qwen_vl_utils.process_vision_info`` and forwarded to the image
        # processor — they cap the visual-token count so a high-DPI page
        # doesn't blow past `max_model_len`.
        messages: List[Dict[str, Any]] = []
        if self._configuration.system_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": self._configuration.system_prompt,
                }
            )
        image_content: Dict[str, Any] = {"type": "image", "image": image}
        if self._configuration.image_max_pixels is not None:
            image_content["max_pixels"] = self._configuration.image_max_pixels
        if self._configuration.image_min_pixels is not None:
            image_content["min_pixels"] = self._configuration.image_min_pixels
        messages.append(
            {
                "role": "user",
                "content": [
                    image_content,
                    {"type": "text", "text": prompt},
                ],
            }
        )
        return messages

    def _build_vllm_inputs(
        self,
        messages: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        # Two-step: render the prompt text from the chat template, then run
        # qwen_vl_utils to extract resized image tensors at Qwen3-VL's expected
        # pixel budget.
        from qwen_vl_utils import process_vision_info

        prompt_text = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)

        multi_modal_data: Dict[str, Any] = {}
        if image_inputs:
            multi_modal_data["image"] = image_inputs
        if video_inputs:
            multi_modal_data["video"] = video_inputs

        inputs: Dict[str, Any] = {"prompt": prompt_text}
        if multi_modal_data:
            inputs["multi_modal_data"] = multi_modal_data
        return inputs

    def generate(
        self,
        image: Image.Image,
        prompt: str,
        sampling_overrides: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Single image+prompt → generated text."""
        self._ensure_loaded("generate")
        messages = self._build_messages(image, prompt)
        vllm_inputs = self._build_vllm_inputs(messages)
        sampling_params = self._build_sampling_params(sampling_overrides)
        outputs = self._llm.generate(
            [vllm_inputs],
            sampling_params=sampling_params,
            use_tqdm=False,
        )
        return outputs[0].outputs[0].text

    def generate_batch(
        self,
        items: Sequence[Dict[str, Any]],
        sampling_overrides: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """Batched variant: each item is ``{"image": PIL.Image, "prompt": str}``.

        vLLM batches multimodal prompts natively when given a list of
        per-prompt input dicts. Order of returns matches order of items.
        """
        self._ensure_loaded("generate_batch")
        all_inputs: List[Dict[str, Any]] = []
        for item in items:
            messages = self._build_messages(item["image"], item["prompt"])
            all_inputs.append(self._build_vllm_inputs(messages))
        sampling_params = self._build_sampling_params(sampling_overrides)
        outputs = self._llm.generate(
            all_inputs,
            sampling_params=sampling_params,
            use_tqdm=False,
        )
        return [output.outputs[0].text for output in outputs]

    def release(self) -> None:
        self._llm = None
        self._processor = None
        self._loaded = False
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
