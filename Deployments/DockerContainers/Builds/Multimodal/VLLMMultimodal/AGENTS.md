# AGENTS.md — VLLMMultimodal pickup guide

You are an AI coding agent (Claude Code, Codex, OpenClaw, or other) pointed at this build. This file plus `STATUS.md` tells you everything you need to continue the work.

## Read in this order

1. **`STATUS.md`** (this directory) — full phase state, build artifacts, decision rationale. Source of truth for *what's been done and why*.
2. **This file** — concrete commands for environment verification, smoke tests, rebuild, and common pitfalls.
3. **`NEXT_STEPS.md`** (this directory) — open backlog items, each sized for a single session.
4. **`README.md`** (this directory) — build/run mechanics in isolation.

If the user asks you to "continue Phase 2 work" or "pick up the multimodal build" or similar, start here.

## What this is, in one paragraph

A single Docker image (`vllm-multimodal:25.06-py3`) hosts three vision-language workloads on the user's RTX 3060 (12 GB Ampere, sm_86): **(1)** MinerU2.5-Pro for structured PDF document extraction; **(2)** Qwen3-VL-4B AWQ-8bit for general-purpose VLM Q&A on images; **(3)** ColQwen2.5-v0.2 for multi-vector retrieval against page images. All three share the same vLLM 0.11.2 / torch 2.9.0 / transformers 4.57.6 stack. The end-use case is parsing propulsion P&IDs (example PDF corpus).

## Repo locations

```
/home/propdev/.openclaw/workspace/workspace2/repos/InServiceOfX/
  Deployments/DockerContainers/Builds/Multimodal/VLLMMultimodal/   # THIS dir
  PythonLibraries/HuggingFace/MoreMinerU/                          # wrappers + tests
  PythonApplications/CLIPDFExtraction/                             # MinerU CLI (Phase 1)
  PythonApplications/CLIPDFQwen3VLChat/                            # Qwen3-VL CLI (Phase 2)
  Scripts/QuickAliases/QuickDockerBuilder.py                       # build/run wrapper
```

## Environment verification

Before doing anything else, sanity-check the host environment:

```bash
# (1) Image is present and matches the build config:
docker images vllm-multimodal:25.06-py3
# Expect a row with size ~36-40 GB. If absent, rebuild — see below.

# (2) Model weights on disk:
ls /media/propdev/9dc1a908-7eff-4e1c-8231-ext4/home/propdev/Data/Models/Multimodal/
# Expect at minimum:
#   opendatalab/MinerU2.5-Pro-2604-1.2B/
#   cyankiwi/Qwen3-VL-4B-Instruct-AWQ-8bit/
#   vidore/colqwen2.5-v0.2/                  (LoRA only; base auto-fetches to HF cache)

# (3) HF cache for ColQwen's base:
ls /media/propdev/9dc1a908-7eff-4e1c-8231-ext4/home/propdev/Data/.cache/huggingface/
# After first ColQwen smoke this should contain hub/models--vidore--colqwen2.5-base/...

# (4) GPUs:
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
# Expect GPU 1 = RTX 3060 12 GB. Phases 2 + 3 are too tight for GPU 0 (980 Ti, 6 GB).
```

## Per-phase smoke tests

Each smoke test runs end-to-end inside a fresh container. The container is launched via `QuickDockerBuilder.py`, which reads `run_configuration.yml` for volumes (model dir, repo, HF cache). Run from the host:

```bash
python3 /home/propdev/.openclaw/workspace/workspace2/repos/InServiceOfX/Scripts/QuickAliases/QuickDockerBuilder.py \
  run Multimodal/VLLMMultimodal --gpu-id 1 --entrypoint /bin/bash
```

This drops you into a bash shell inside the container as `root@<containerid>:/`. Then:

### Phase 1 — MinerU2.5-Pro PDF → structured JSON

```bash
cd /InServiceOfX/PythonApplications/CLIPDFExtraction
cp Configurations/mineru_configuration.yml.example      Configurations/mineru_configuration.yml
cp Configurations/pdf_extraction_configuration.yml.example  Configurations/pdf_extraction_configuration.yml
python Executables/main_CLIPDFExtraction.py --currentpath
```

Expected: per-PDF subdir under `/Workspace/Generated/CLIPDFExtraction/<pdf_stem>/` with `page_N.md` (structured per-element JSON), optional `page_N.png`, and `manifest.json`. Each page ~8-30 s on the 3060.

### Phase 2 — Qwen3-VL-4B AWQ-8bit chat about a PDF page

```bash
# (a) Wrapper-level smoke (single image + one prompt):
python /InServiceOfX/PythonLibraries/HuggingFace/MoreMinerU/tests/smoke_qwen3vl_gpu.py

# (b) Full PDF CLI (per-page freeform response):
cd /InServiceOfX/PythonApplications/CLIPDFQwen3VLChat
cp Configurations/qwen3vl_configuration.yml.example  Configurations/qwen3vl_configuration.yml
cp Configurations/pdf_chat_configuration.yml.example Configurations/pdf_chat_configuration.yml
python Executables/main_CLIPDFQwen3VLChat.py --currentpath
```

Expected: a paragraph-length response describing the page (title, revision, owning org, major systems). Per-page latency ~10-12 s.

### Phase 3 — ColQwen2.5 multi-vector retrieval

```bash
python /InServiceOfX/PythonLibraries/HuggingFace/MoreMinerU/tests/smoke_colqwen_gpu.py
```

On first run inside a new HF cache, `transformers` downloads `vidore/colqwen2.5-base` (~8 GB, ~90 s on fast link). Subsequent runs use the cache. Output is a 3×3 MaxSim score matrix printed to stdout — strong diagonal-ish hits prove retrieval works.

## How to rebuild the image

If `build_configuration.yml`, any `Dockerfile.*` component, or any `*.yml.example` changes:

```bash
python3 /home/propdev/.openclaw/workspace/workspace2/repos/InServiceOfX/Scripts/QuickAliases/QuickDockerBuilder.py \
  build Multimodal/VLLMMultimodal
```

Cached layers stay valid — only changed component layers and downstream layers re-run. The full build from scratch is ~10-20 min (mostly torch + CUDA wheel downloads).

## Development loop (no rebuild needed)

The InServiceOfX repo is **bind-mounted** at `/InServiceOfX` inside the container. You can edit code on the host (in any editor) and re-run inside the container without rebuilding. The Docker image only needs rebuilding when `Dockerfile.*` / `build_configuration.yml` changes.

## Common pitfalls — agents have hit these before

| Symptom | Cause | Fix |
|---|---|---|
| `AttributeError: Qwen2Tokenizer has no attribute all_special_tokens_extended` | transformers 5.x silently installed (vLLM 0.11.2 requires 4.x) | The Dockerfile already pins `transformers==4.57.6`. Don't reorder layers. |
| `ImportError: ... flash_attn_2_cuda.so: undefined symbol: _ZN3c104cuda9SetDeviceEa` | NV-shipped flash-attn compiled against alpha torch | Already handled — `Dockerfile.pytorch_reinstall` uninstalls flash-attn; vLLM falls back to xformers. |
| `ResolutionImpossible ... nvidia-cudnn-frontend==1.12.0 ... >=1.13.0` | NV constraint file pins old cudnn-frontend; vLLM 0.11.1+ needs newer | Already handled — `Dockerfile.pytorch_reinstall` strips the constraint. |
| `ImportError: cannot import name 'ColQwen2_5' from 'colpali_engine.models'` | colpali-engine < 0.3.9 | Pin must be in `[0.3.9, 0.3.12]` — see STATUS.md decision. |
| `ImportError: cannot import name 'ModernVBertModel' from 'transformers'` | colpali-engine ≥ 0.3.13 eager-imports a transformers 5.x symbol | Pin must be ≤ 0.3.12 — see STATUS.md decision. |
| `ImportError: Found an incompatible version of torchao` | peft 0.16+ raises on torchao < 0.16; NV ships 0.11.0+git | Already handled — `Dockerfile.colqwen` uninstalls torchao. |
| `ValueError: User-specified max_model_len (16384) is greater than the derived max_model_len (8192...)` | MinerU2.5-Pro is Qwen2VL-1.2B based, capped at 8192 | Already set in `mineru_configuration.yml.example`; do not raise. |
| `ValueError: To serve at least one request with the models's max seq len (8192), (1.12 GiB KV cache is needed...)` | Qwen3-VL AWQ-8bit + max_model_len=8192 needs `gpu_memory_utilization ≥ 0.95` | Bump it. 0.90 leaves the KV cache 20 MB short. |
| `ValueError: The decoder prompt (length 11466) is longer than the maximum model length of 8192` | Qwen3-VL: high-DPI rasterized P&ID page = ~15k visual tokens | Set `image_max_pixels: 1003520` in `qwen3vl_configuration.yml`. The wrapper threads it through to `qwen_vl_utils.process_vision_info`. |
| `torch.OutOfMemoryError: ... Tried to allocate 2.89 GiB. GPU has ... 1.78 GiB free` | Qwen3-VL-4B bf16 — doesn't fit on 12 GB Ampere | Use AWQ-8bit (`cyankiwi/Qwen3-VL-4B-Instruct-AWQ-8bit`). FP8 won't work on Ampere (sm_86 < sm_89). |

## Git policy

The user has explicit rules:

- **NEVER commit or push to `master` / `main`.** The user merges manually. Work on feature branches and push those.
- Current branch is `feat/vllm-multimodal-mineru` (pushed to origin).
- All multimodal work this session lives there — see `STATUS.md` for the commit list.
- Don't squash or rewrite history; commits are individually useful as a paper trail of decisions.

## When to stop and ask the user

- Any GPU configuration change that risks impacting their other work (don't bump `--gpu-id 0` without asking).
- Any HF download larger than ~10 GB (the ColQwen base was already authorized; new model downloads should be confirmed).
- Anything that would force a `docker build --no-cache` (full rebuild from scratch is ~20 min).
- If you can't get a smoke test to pass after 3 attempts — capture the actual error in `STATUS.md` and stop.

## Where the live (gitignored) configs live

These four YAMLs are **gitignored** (each app's `Configurations/`) — they are user-customized copies of `.example` files. If you need to inspect or edit them, look at them on disk; don't expect them in `git ls-files`:

```
Deployments/DockerContainers/Builds/Multimodal/VLLMMultimodal/build_configuration.yml
Deployments/DockerContainers/Builds/Multimodal/VLLMMultimodal/run_configuration.yml
PythonApplications/CLIPDFExtraction/Configurations/mineru_configuration.yml
PythonApplications/CLIPDFExtraction/Configurations/pdf_extraction_configuration.yml
PythonApplications/CLIPDFQwen3VLChat/Configurations/qwen3vl_configuration.yml
PythonApplications/CLIPDFQwen3VLChat/Configurations/pdf_chat_configuration.yml
```

The `.example` versions ARE tracked and reflect the recommended defaults.

## Unit tests

The library has lightweight (no-GPU, no-vLLM) configuration tests:

```bash
# inside the container:
cd /InServiceOfX/PythonLibraries/HuggingFace/MoreMinerU
python -m pytest tests/ -q
# Expect: 19 passed (5 MinerU + 8 Qwen3VL + 6 ColQwen)
```

The `smoke_*` tests are GPU-only and not collected by pytest (filename prefix doesn't start with `test_`).
