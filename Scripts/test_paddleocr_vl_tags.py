"""
Quick test: PaddleOCR-VL-1.5 tag extraction vs Tesseract baseline on rev09 page 2.

PaddleOCR-VL-1.5 officially requires transformers>=5.0 for the transformers path.
This script tries two fallbacks in order:
  1. AutoModelForImageTextToText with trust_remote_code=True (may work on 4.x)
  2. AutoModelForCausalLM with trust_remote_code=True

Run inside the vllm-multimodal container:
    python /InServiceOfX/Scripts/test_paddleocr_vl_tags.py
"""

import json
import re
import time
from pathlib import Path

import torch
from PIL import Image

MODEL_PATH = "/Data/Models/Multimodal/PaddlePaddle/PaddleOCR-VL-1.5"
IMAGE_PATH = (
    "/Workspace/Generated/CLIPDFExtraction"
    "/psas_pid-20/page_1.png"
)
TESSERACT_JSON = (
    "/Workspace/Generated/CLIPDFTesseractExtraction"
    "/psas_pid-20/page_1.json"
)

TAG_PATTERN = re.compile(r"\b[A-Z][A-Z0-9\-]{3,}\b")

PROMPTS = {
    "ocr": "OCR:",
    "spotting": "Text Spotting:",
}


def load_tesseract_tags() -> set[str]:
    data = json.loads(Path(TESSERACT_JSON).read_text())
    return set(data.get("merged_tags", []))


def extract_tags(text: str) -> set[str]:
    return set(TAG_PATTERN.findall(text))


def score(found: set[str], tesseract: set[str], label: str) -> None:
    tp = found & tesseract
    extra = found - tesseract
    missed = tesseract - found
    print(f"\n  [{label}]")
    print(f"  Tags found:          {len(found)}")
    print(f"  Overlap w Tesseract: {len(tp)}")
    print(f"  Extra:               {len(extra)}  {sorted(extra)[:8]}")
    print(f"  Missed:              {len(missed)}  {sorted(missed)[:8]}")
    if tesseract:
        print(f"  Recall vs Tesseract: {len(tp)/len(tesseract)*100:.1f}%")


def try_transformers(image: Image.Image, tesseract_tags: set[str]) -> bool:
    """Try loading via transformers (requires >=5.0 officially; may fail on 4.x)."""
    print("\nAttempting transformers path (trust_remote_code=True)...")
    try:
        from transformers import AutoModelForImageTextToText, AutoProcessor
        import transformers
        print(f"  transformers version: {transformers.__version__}")

        processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        print(f"  Model loaded. device={next(model.parameters()).device}")
    except Exception as e:
        print(f"  FAILED to load: {e}")
        return False

    for task, prompt in PROMPTS.items():
        try:
            # Build minimal chat input — PaddleOCR-VL uses same apply_chat_template pattern
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            ).to(model.device)
            inputs.pop("token_type_ids", None)

            t0 = time.time()
            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=4096)
            elapsed = time.time() - t0

            out = processor.decode(
                generated_ids[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )
            print(f"\n  prompt={prompt!r}  elapsed={elapsed:.1f}s")
            print(f"  output[:400]: {out[:400]!r}")
            score(extract_tags(out), tesseract_tags, task)
        except Exception as e:
            print(f"  inference error for {task}: {e}")

    return True


def main():
    tesseract_tags = load_tesseract_tags()
    print(f"Tesseract baseline: {len(tesseract_tags)} tags")
    print(f"Image: {IMAGE_PATH}\n")

    image = Image.open(IMAGE_PATH).convert("RGB")
    print(f"Image size: {image.size}")

    success = try_transformers(image, tesseract_tags)
    if not success:
        print("\n" + "="*60)
        print("PaddleOCR-VL-1.5 could not load under current transformers version.")
        print("Options to test it:")
        print("  1. pip install 'transformers>=5.0.0'  (breaks vLLM in this container)")
        print("  2. Create a fresh venv:  uv venv /tmp/paddle-test && uv pip install transformers>=5.0 pillow torch")
        print("  3. Use the PaddleOCR genai vLLM Docker image instead of this one")
        print("  4. Skip — if GLM-OCR results are good, PaddleOCR-VL-1.5 adds no new capability")


if __name__ == "__main__":
    main()
