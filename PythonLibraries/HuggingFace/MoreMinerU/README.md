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

`generate(image, prompt)` returns natural-language text. Unlike MinerU, this is open-ended and not structured. Defaults to greedy decoding (`temperature=0`, `max_tokens=1024`); override per call via `sampling_overrides=` or globally via `default_sampling_params` in the YAML.

## Memory budget notes

| Model | Weights (bf16) | Comfortable `gpu_memory_utilization` on 12 GB / 8 GB | Comfortable `max_model_len` |
| --- | --- | --- | --- |
| MinerU2.5-Pro (1.2B) | ~2.4 GB | 0.85 / 0.80 | 8192 (architectural cap; do not raise) |
| Qwen3-VL-4B | ~8 GB | doesn't fit on 12 GB in bf16 (see below) / never | n/a in bf16 on consumer GPUs |

### Qwen3-VL-4B in bf16 doesn't fit on a 12 GB GPU under vLLM 0.11.2

Empirically reproduced on RTX 3060 (11.63 GiB visible): vLLM's `profile_run` at engine init unconditionally tries to allocate a ~2.89 GiB activation buffer (Qwen2VL-style multimodal forward pass through the vision tower). Weights load uses ~9.1 GB of the 11.63 GiB physical capacity (`gpu_memory_utilization` not the cause — the model genuinely uses that much), leaving only ~1.78 GB free, less than the 2.89 GB the profiler needs.

`max_model_len` does not affect this: the same 2.89 GiB error reproduces at 2048, 4096, and 8192. `max_num_seqs=1` and `limit_mm_per_prompt={"image": 1}` did not reduce the profile allocation either.

Workarounds (untested locally; user-side decision):
- **AWQ/INT4 quantized weights** (e.g. `Qwen/Qwen3-VL-4B-Instruct-AWQ` if released by Qwen team — check HF) drop weights from ~8 GB to ~3 GB; should fit comfortably with KV cache.
- **Run on a 16 GB+ GPU** (3070 Ti / 4070 Ti / A4000 / 3090 etc.).
- **Use a smaller VLM** like Qwen2.5-VL-3B-Instruct in bf16 (~6 GB weights), or InternVL/Phi-3.5-Vision in similar size class.

The `Qwen3VLVLLM` wrapper itself is fully implemented and unit-tested; the constraint is purely about GPU memory at validation time.
