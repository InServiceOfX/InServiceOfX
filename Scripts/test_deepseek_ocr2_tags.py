"""
DeepSeek-OCR-2 P&ID tag extraction test.

Compares DeepSeek-OCR-2 (vLLM, "Free OCR" mode with crop tiling) against
Tesseract baseline on a reference P&ID page.

Run inside the vllm-multimodal container (after rebuild):
    python /InServiceOfX/Scripts/test_deepseek_ocr2_tags.py

PYTHONPATH must include the DeepSeek-OCR-2 vLLM scripts dir — set by the
Dockerfile.deepseek_ocr2 layer via ENV PYTHONPATH.
"""

import json
import os
import re
import sys
import time
from pathlib import Path

# vLLM 0.11.2 has no V0 engine — AsyncLLMEngine is always V1.
# VLLM_ENABLE_V1_MULTIPROCESSING=0 makes the V1 EngineCore run in-process
# (no subprocess spawn), so our import shims apply to model loading too.
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
# Use GPU 0 visible to the container (GPU 1 on host when launched with --gpu-id 1).
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from PIL import Image, ImageOps

# --- vLLM 0.11.2 compatibility shims for deepseek_ocr2.py ---
# deepseek_ocr2.py was written against an older vLLM API. Two symbols moved:
#   1. SamplingMetadata: only used as a type annotation in deepseek_ocr2.py;
#      stub with an empty class so the import resolves without pulling V1 engine.
#   2. set_default_torch_dtype: moved to vllm.utils.torch_utils in 0.11.2;
#      was previously exported from vllm.model_executor.model_loader.utils.
import vllm.model_executor as _me
import vllm.model_executor.model_loader.utils as _mlu
from vllm.utils.torch_utils import set_default_torch_dtype as _sdtd
_me.SamplingMetadata = type("SamplingMetadata", (), {})
_mlu.set_default_torch_dtype = _sdtd

# Stub flash_attn: uninstalled (ABI-incompatible with stable torch 2.9.0).
# deepencoderv2/sam_vary_sdpa.py imports flash_attn_qkvpacked_func at module
# level but the actual call is commented out — PyTorch SDPA is used instead.
import types as _types
_flash_attn_stub = _types.ModuleType("flash_attn")
_flash_attn_stub.flash_attn_qkvpacked_func = lambda *args, **kwargs: None
sys.modules["flash_attn"] = _flash_attn_stub

MODEL_PATH = "/Data/Models/Multimodal/deepseek-ai/DeepSeek-OCR-2"
IMAGE_PATH = (
    "/Workspace/Generated/CLIPDFExtraction"
    "/psas_pid-20/page_1.png"
)
TESSERACT_JSON = (
    "/Workspace/Generated/CLIPDFTesseractExtraction"
    "/psas_pid-20/page_1.json"
)
OUTPUT_DIR = "/Workspace/Generated/DeepSeekOCR2Test/psas-page1"

CROP_MODE = True

# P&ID compound tags: uppercase letter + 3+ uppercase/digit/hyphen chars.
TAG_PATTERN = re.compile(r"\b[A-Z][A-Z0-9\-]{3,}\b")


def load_tesseract_tags() -> set[str]:
    data = json.loads(Path(TESSERACT_JSON).read_text())
    return set(data.get("merged_tags", []))


def extract_tags(text: str) -> set[str]:
    return set(TAG_PATTERN.findall(text))


def score(found: set[str], tesseract: set[str], label: str) -> None:
    tp = found & tesseract
    extra = found - tesseract
    missed = tesseract - found
    print(f"\n{'='*50}")
    print(f"  Prompt mode: {label}")
    print(f"  Tags found by model:    {len(found)}")
    print(f"  Tesseract baseline:     {len(tesseract)}")
    print(f"  Overlap (TP):           {len(tp)}")
    print(f"  Extra vs Tesseract:     {len(extra)}  {sorted(extra)[:10]}")
    print(f"  Missed vs Tesseract:    {len(missed)}  {sorted(missed)[:10]}")
    if tesseract:
        print(f"  Recall vs Tesseract:    {len(tp)/len(tesseract)*100:.1f}%")
    if found:
        print(f"  Precision vs Tesseract: {len(tp)/len(found)*100:.1f}%")


def build_engine():
    from vllm import LLM, SamplingParams
    from vllm.model_executor.models.registry import ModelRegistry

    # Use the "module:class" string form so vLLM can lazily import the class
    # in the EngineCore subprocess (passing a class object directly can't be
    # serialized across the multiprocessing boundary in V1).
    ModelRegistry.register_model("DeepseekOCR2ForCausalLM", "deepseek_ocr2:DeepseekOCR2ForCausalLM")

    print("Building LLM engine (V0 synchronous)...", flush=True)
    llm = LLM(
        model=MODEL_PATH,
        hf_overrides={"architectures": ["DeepseekOCR2ForCausalLM"]},
        dtype="bfloat16",
        max_model_len=8192,
        # enforce_eager skips CUDA graph capture — faster startup, avoids hangs.
        enforce_eager=True,
        trust_remote_code=True,
        tensor_parallel_size=1,
        # 6.4 GB weights on 12 GB GPU; 0.90 leaves ~4.4 GB for KV cache.
        gpu_memory_utilization=0.90,
    )
    print("Engine ready.", flush=True)
    return llm


def run_inference(llm, image_features, prompt: str) -> str:
    from vllm import SamplingParams
    from process.ngram_norepeat import NoRepeatNGramLogitsProcessor

    logits_processors = [
        NoRepeatNGramLogitsProcessor(
            ngram_size=20,
            window_size=90,
            whitelist_token_ids={128821, 128822},
        )
    ]
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=8192,
        logits_processors=logits_processors,
        skip_special_tokens=False,
    )

    if image_features and "<image>" in prompt:
        inputs = {"prompt": prompt, "multi_modal_data": {"image": image_features}}
    else:
        inputs = {"prompt": prompt}

    outputs = llm.generate(inputs, sampling_params)
    return outputs[0].outputs[0].text


def main() -> None:
    tesseract_tags = load_tesseract_tags()
    print(f"Tesseract baseline: {len(tesseract_tags)} tags")
    print(f"Image: {IMAGE_PATH}")

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    image = ImageOps.exif_transpose(Image.open(IMAGE_PATH).convert("RGB"))
    print(f"Image size: {image.size}")

    from process.image_process import DeepseekOCR2Processor

    # Pre-process image once; reuse for both prompts.
    t0 = time.time()
    image_features = DeepseekOCR2Processor().tokenize_with_images(
        images=[image], bos=True, eos=True, cropping=CROP_MODE
    )
    print(f"Image preprocessing: {time.time() - t0:.1f}s", flush=True)

    # Build engine once; run both prompts against it.
    llm = build_engine()

    prompts = {
        "free_ocr": "<image>\nFree OCR.",
        "grounding_markdown": "<image>\n<|grounding|>Convert the document to markdown.",
    }

    for mode, prompt in prompts.items():
        print(f"\n{'='*50}")
        print(f"Running mode: {mode!r}")
        print(f"Prompt: {prompt!r}", flush=True)

        t0 = time.time()
        raw_output = run_inference(llm, image_features, prompt)
        elapsed_infer = time.time() - t0
        print(f"Inference: {elapsed_infer:.1f}s")

        out_file = Path(OUTPUT_DIR) / f"result_{mode}.txt"
        out_file.write_text(raw_output)
        print(f"Raw output saved to: {out_file}")
        print(f"Output preview (first 500 chars): {raw_output[:500]!r}")

        found_tags = extract_tags(raw_output)
        score(found_tags, tesseract_tags, mode)


if __name__ == "__main__":
    main()
