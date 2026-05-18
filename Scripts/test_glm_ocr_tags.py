"""
Quick test: GLM-OCR tag extraction vs Tesseract baseline on a reference P&ID page.

Run inside the vllm-multimodal container:
    python /InServiceOfX/Scripts/test_glm_ocr_tags.py
"""

import json
import re
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL_PATH = "/Data/Models/Multimodal/zai-org/GLM-OCR"
IMAGE_PATH = (
    "/Workspace/Generated/CLIPDFExtraction"
    "/psas_pid-20/page_1.png"
)
TESSERACT_JSON = (
    "/Workspace/Generated/CLIPDFTesseractExtraction"
    "/psas_pid-20/page_1.json"
)

# P&ID component tags: uppercase alphanumeric, often hyphen-separated, 4+ chars
TAG_PATTERN = re.compile(r"\b[A-Z][A-Z0-9\-]{3,}\b")


def load_tesseract_tags() -> set[str]:
    data = json.loads(Path(TESSERACT_JSON).read_text())
    return set(data.get("merged_tags", []))


def run_glm_ocr(image: Image.Image, prompt_text: str, processor, model) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": image},
                {"type": "text", "text": prompt_text},
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
        generated_ids = model.generate(**inputs, max_new_tokens=8192)
    elapsed = time.time() - t0

    output_text = processor.decode(
        generated_ids[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=False,
    )
    return output_text, elapsed


def extract_tags(text: str) -> set[str]:
    return set(TAG_PATTERN.findall(text))


def score(found: set[str], tesseract: set[str]) -> None:
    tp = found & tesseract
    extra = found - tesseract
    missed = tesseract - found
    print(f"  Tags found by model:    {len(found)}")
    print(f"  Tesseract baseline:     {len(tesseract)}")
    print(f"  Overlap (TP):           {len(tp)}")
    print(f"  Extra vs Tesseract:     {len(extra)}  {sorted(extra)[:10]}")
    print(f"  Missed vs Tesseract:    {len(missed)}  {sorted(missed)[:10]}")
    if len(tesseract):
        print(f"  Recall vs Tesseract:    {len(tp)/len(tesseract)*100:.1f}%")


def main():
    tesseract_tags = load_tesseract_tags()
    print(f"Tesseract baseline: {len(tesseract_tags)} tags\n")

    print("Loading GLM-OCR model...")
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, device_map="auto"
    )
    print(f"Loaded in {time.time()-t0:.1f}s  device={next(model.parameters()).device}\n")

    image = Image.open(IMAGE_PATH).convert("RGB")
    print(f"Image: {IMAGE_PATH}  size={image.size}")

    # --- Run 1: plain text dump ---
    print("\n--- Run 1: 'Text Recognition:' ---")
    raw_text, elapsed = run_glm_ocr(image, "Text Recognition:", processor, model)
    print(f"  Elapsed: {elapsed:.1f}s")
    print(f"  Output (first 400 chars): {raw_text[:400]!r}")
    found = extract_tags(raw_text)
    score(found, tesseract_tags)

    # --- Run 2: structured JSON extraction ---
    json_prompt = (
        'Please output the information in the following JSON format:\n'
        '{"component_tags": []}\n'
        'Extract all instrument and component identifier tags visible in this P&ID diagram '
        '(e.g. S2BHFB, S2OTNK, RP-1). Output only the JSON.'
    )
    print("\n--- Run 2: JSON extraction ---")
    raw_json, elapsed = run_glm_ocr(image, json_prompt, processor, model)
    print(f"  Elapsed: {elapsed:.1f}s")
    print(f"  Output (first 400 chars): {raw_json[:400]!r}")

    # Try to parse JSON from output
    json_match = re.search(r'\{.*\}', raw_json, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group())
            json_tags = set(parsed.get("component_tags", []))
            print(f"  Parsed {len(json_tags)} tags from JSON")
            score(json_tags, tesseract_tags)
        except json.JSONDecodeError as e:
            print(f"  JSON parse error: {e}")
            found_json = extract_tags(raw_json)
            score(found_json, tesseract_tags)
    else:
        print("  No JSON block found — falling back to regex")
        found_json = extract_tags(raw_json)
        score(found_json, tesseract_tags)


if __name__ == "__main__":
    main()
