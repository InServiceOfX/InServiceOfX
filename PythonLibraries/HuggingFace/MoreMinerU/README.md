# MoreMinerU

Thin Python wrappers for serving vLLM-based multimodal models in-process. Started as a wrapper around [`mineru-vl-utils`](https://pypi.org/project/mineru-vl-utils/) for `opendatalab/MinerU2.5-Pro-2604-1.2B`; now also hosts a wrapper for `Qwen/Qwen3-VL-4B-Instruct`. The two share the same vLLM engine and Docker image, so they live in the same library despite the name.

Designed to run inside the `VLLMMultimodal` Docker image (see `Deployments/DockerContainers/Builds/Multimodal/VLLMMultimodal/`). Configuration is YAML-driven via Pydantic configuration classes — no hardcoded paths.

## Layout

```
moremineru/
  Configurations/
    MinerUConfiguration.py     # Pydantic config for MinerU2.5-Pro
    Qwen3VLConfiguration.py    # Pydantic config for Qwen3-VL family
  Applications/
    MinerU2_5ProVLLM.py        # Wraps vllm.LLM + MinerUClient (two-step extraction)
    Qwen3VLVLLM.py             # Wraps vllm.LLM directly (chat with image+prompt)
```

## MinerU2.5-Pro — structured document extraction

```python
from pathlib import Path
from PIL import Image
from moremineru.Configurations import MinerUConfiguration
from moremineru.Applications import MinerU2_5ProVLLM

config = MinerUConfiguration.from_yaml(Path("/path/to/mineru_configuration.yml"))
runner = MinerU2_5ProVLLM(config)
runner.load()
result = runner.extract_from_image(Image.open("/path/to/page.png"))
print(result)  # list of {type, bbox, content} dicts
runner.release()
```

`extract_from_image` returns structured per-element output (headers, tables as HTML, equations as LaTeX, figures with captions). The CLI in `PythonApplications/CLIPDFExtraction/` drives this end-to-end on a directory of PDFs.

## Qwen3-VL — general-purpose VLM chat

Wraps `vllm.LLM` following the QwenLM/Qwen3-VL upstream's canonical pattern: `processor.apply_chat_template` + `qwen_vl_utils.process_vision_info` + `llm.generate({"prompt": ..., "multi_modal_data": {"image": [...]}})`. The Docker image installs `qwen-vl-utils==0.0.14` and `accelerate` for this path.

```python
from pathlib import Path
from PIL import Image
from moremineru.Configurations import Qwen3VLConfiguration
from moremineru.Applications import Qwen3VLVLLM

config = Qwen3VLConfiguration.from_yaml(Path("/path/to/qwen3vl_configuration.yml"))
runner = Qwen3VLVLLM(config)
runner.load()
image = Image.open("/path/to/page.png")
answer = runner.generate(image, "Describe the major systems shown in this P&ID.")
print(answer)
# Batched variant:
answers = runner.generate_batch([
    {"image": img1, "prompt": "What's the title?"},
    {"image": img2, "prompt": "List all visible components."},
])
runner.release()
```

`generate(image, prompt)` returns natural-language text. Defaults match the Qwen3-VL model card's recommended *VL* sampling: `top_p=0.8`, `top_k=20`, `temperature=0.7`, `presence_penalty=1.5`, `repetition_penalty=1.0`, `max_tokens=1024`. Override per call via `sampling_overrides={...}` or globally via `default_sampling_params` in the YAML. For grounded structured extraction, pass `sampling_overrides={"temperature": 0.0}` to force greedy.

## Memory budget notes

| Model | Variant we use | Weights on disk | Fits 12 GB? | Notes |
| --- | --- | --- | --- | --- |
| MinerU2.5-Pro (1.2B) | bf16 (full) | ~2.4 GB | yes | `max_model_len 8192` is the model's architectural cap |
| Qwen3-VL-4B | AWQ-8bit (`cyankiwi/Qwen3-VL-4B-Instruct-AWQ-8bit`) | ~4 GB | yes | Pass `quantization="compressed-tensors"` to vLLM — that's the AWQ-8bit packaging the cyankiwi build uses (via `llm-compressor`) |

### Why AWQ-8bit, not bf16 or FP8

- **bf16 (`Qwen/Qwen3-VL-4B-Instruct`):** ~8 GB weights + ~2.89 GB profile-run activation buffer at engine init = 12+ GB peak. Confirmed not to fit on a 12 GB 3060 under vLLM 0.11.2 — `max_model_len`, `max_num_seqs`, and `limit_mm_per_prompt` don't reduce the profile allocation.
- **FP8 (`Qwen/Qwen3-VL-4B-Instruct-FP8`):** halves weights but FP8 hardware acceleration in vLLM requires **compute capability ≥ 8.9** (Ada Lovelace / Hopper). Ampere GPUs (sm_86, including the 3060) can't run it.
- **AWQ-4bit (`cyankiwi/Qwen3-VL-4B-Instruct-AWQ-4bit`):** smallest (~2-3 GB), works on Ampere via vLLM AWQ-Marlin kernel. Choose this if 8-bit doesn't fit in your particular workload's KV cache budget.
- **AWQ-8bit (`cyankiwi/Qwen3-VL-4B-Instruct-AWQ-8bit`):** best quality among Ampere-compatible options. Default in this repo.
