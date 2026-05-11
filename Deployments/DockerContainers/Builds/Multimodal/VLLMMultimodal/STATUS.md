# VLLMMultimodal — Status

> **Agents: read [`AGENTS.md`](./AGENTS.md) first for pickup commands, then this file for state and decisions, then [`NEXT_STEPS.md`](./NEXT_STEPS.md) for open backlog.** This file is the source of truth for *what's been done and why*, not for *what to do next*.

Last updated: 2026-05-10 (Phase 3 smoke-tested).

---

## TL;DR for an incoming agent

- **All 3 phases smoke-tested end-to-end** on the user's RTX 3060 (12 GB Ampere) as of 2026-05-10:
  - **Phase 1**: MinerU2.5-Pro for structured PDF document extraction → `CLIPDFExtraction` CLI ran the full 12-PDF example P&IDs corpus.
  - **Phase 2**: Qwen3-VL-4B AWQ-8bit for general-purpose VLM Q&A → `CLIPDFQwen3VLChat` CLI ran rev11 (7 pages, ~10 s/page).
  - **Phase 3**: ColQwen2.5-v0.2 multi-vector retrieval → wrapper smoke produced a sensible MaxSim score matrix.
- The current image (`vllm-multimodal:25.06-py3` sha `f96c8b89ae50`, ~37 GB) hosts all three. torch 2.9.0+cu128, vllm 0.11.2, transformers 4.57.6.
- All design decisions below are load-bearing — don't change base image, torch reinstall, flash-attn uninstall, cudnn-frontend strip, transformers 4.x pin, colpali-engine version pin, or torchao uninstall without reading the matching "Decisions" entry.
- Open work: indexing + query CLIs for ColQwen retrieval — see [`NEXT_STEPS.md`](./NEXT_STEPS.md).

## Phase status

| Phase | Model | Weights on disk | Wrapper code | CLI wiring | Image built | Smoke-tested |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `opendatalab/MinerU2.5-Pro-2604-1.2B` (1.2B, doc extraction) | yes | yes (`MoreMinerU`) | yes (`CLIPDFExtraction`) | **yes** | **yes (2026-05-10)** |
| 2 | `cyankiwi/Qwen3-VL-4B-Instruct-AWQ-8bit` (4B, AWQ-8bit, general VLM) | yes | yes (`MoreMinerU.Qwen3VLVLLM` via `qwen-vl-utils`) | yes (`CLIPDFQwen3VLChat`) | yes (`vllm-multimodal:25.06-py3` sha `ea1221c0b9c5` with `Dockerfile.qwen3vl`) | **yes (2026-05-10)** |
| 3 | `vidore/colqwen2.5-v0.2` (LoRA on `vidore/colqwen2.5-base`, multimodal retrieval) | LoRA + base (auto-fetched to HF cache) | yes (`MoreMinerU.ColQwen2_5Embedder` via colpali-engine) | not yet — embedding-CLI is a separate Phase 4 | yes (reuses Phase 1 image + new `Dockerfile.colqwen` layer) | **yes (2026-05-10)** |

### Local paths to weights (host)

```
/media/propdev/9dc1a908-7eff-4e1c-8231-ext4/home/propdev/Data/Models/Multimodal/
  opendatalab/MinerU2.5-Pro-2604-1.2B/    # 2.3 GB, full model
  cyankiwi/Qwen3-VL-4B-Instruct-AWQ-8bit/  # ~4 GB AWQ-8bit (replaces deleted bf16)
  vidore/colqwen2.5-v0.2/                 # 240 MB, LoRA adapter only
```

These are mounted into the container at `/Data/Models/Multimodal/...` per `run_configuration.yml.example`.

## Immediate next step

Phase 1 end-to-end smoke test **passed on 2026-05-10** against `example P&IDs rev01.pdf` (3 pages → 3 `page_N.md` JSON files + 3 `page_N.png` rasters + `manifest.json` under `/Workspace/Generated/CLIPDFExtraction/`). MinerU correctly extracted the title page (Acme Aerospace headers, ITAR notice, Change Log table), the P&ID legend (8x4 table with rowspan/colspan preserved), and the components/instrumentation table (with subtotals as `table_footnote` elements). Per-page latencies on RTX 3060 (12 GB), `enforce_eager=true`, xformers attention: 8-30 s/page (text-heavy pages slower).

The smoke test surfaced two image-build issues that have since been baked into the Dockerfile — a clean rebuild now produces a working image without manual fixups (see "Decisions" → transformers pin, flash-attn removal). **The current locally cached image `vllm-multimodal:25.06-py3` (sha256 8d80bfd4299f) does NOT have these fixes baked in** — it's the pre-fix build. Rebuild via `docker_builder build .` to bake them; cached layers up through `Dockerfile.system_deps` stay valid, so the rebuild only redoes the torch / vLLM / mineru layers (~10-15 min).

### Next concrete steps

1. **Rebuild the image** so the fixes are durable: `docker_builder build .` from the build dir (or `python <REPO>/Scripts/QuickAliases/QuickDockerBuilder.py build Multimodal/VLLMMultimodal`).
2. **Full-corpus extraction.** Revert `pdf_extraction_configuration.yml` `input_path` back to the directory (currently pointed at the single rev01 PDF for smoke testing) and re-run to get all 12 example P&IDs revisions. `skip_existing: true` is already set, so re-running is idempotent.
3. **Phase 2 (Qwen3-VL-4B)** wrapper, per the plan in `## Next steps per phase`.

### How to re-run the smoke test (legacy reference)

```bash
# launch interactive shell on GPU 0 (RTX 3060 12 GB or 3070 8 GB)
# — either invoke docker_builder directly:
cd /home/propdev/.openclaw/workspace/workspace2/repos/InServiceOfX/Deployments/DockerContainers/Builds/Multimodal/VLLMMultimodal
../../../../../RustLibraries/docker_builder/target/debug/docker_builder \
  run --build-dir . --gpu-id 0 --entrypoint /bin/bash

# — or via the versioned wrapper (resolves binary + short name automatically;
#   auto-runs `cargo build` if needed; works from anywhere):
python3 <REPO>/Scripts/QuickAliases/QuickDockerBuilder.py \
  run Multimodal/VLLMMultimodal --gpu-id 0 --entrypoint /bin/bash

# inside the container:
cd /InServiceOfX/PythonApplications/CLIPDFExtraction
cp Configurations/mineru_configuration.yml.example   Configurations/mineru_configuration.yml
cp Configurations/pdf_extraction_configuration.yml.example \
   Configurations/pdf_extraction_configuration.yml
# defaults already point at the example PDF corpus dir

python Executables/main_CLIPDFExtraction.py --currentpath
```

Expected output: per-PDF subdir under `/Workspace/Generated/CLIPDFExtraction/<pdf_stem>/` with `page_N.md`, `page_N.png` (if `save_intermediate_images: true`), and `manifest.json`.

`max_model_len: 8192` is set as the default because MinerU2.5-Pro is built on Qwen2VL-1.2B (`max_position_embeddings=8192` per the model's `config.json`). vLLM rejects anything higher — going past the trained range corrupts RoPE positions to NaN. Don't "fix" by setting `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`; that just disables the safety check.

If OOM on the 3070 (8 GB): drop `gpu_memory_utilization` 0.85 → 0.80 and `max_model_len` 8192 → 4096 in `mineru_configuration.yml`.

## Decisions (do not casually reverse)

### Base image: `nvcr.io/nvidia/pytorch:25.06-py3`

- 25.06 is the **last CUDA 12 release** in the NVIDIA PyTorch container line (CUDA 12.9.1, Python 3.12, torch 2.8.0a0).
- 25.07-py3 and later ship CUDA 13. The user has flagged CUDA 13 as breaking nunchaku in a sibling build, and vLLM's CUDA 13 wheel coverage is currently inconsistent.
- The user runs RTX 3060 (12 GB) and 3070 (8 GB) — both sm_86, fully supported on CUDA 12.x. Conservative compat is preferred over newer features.

### Torch reinstall (`Dockerfile.pytorch_reinstall`)

- The NV container ships `torch==2.8.0a0+5228986c39`. The `a0` is a PEP 440 pre-release tag.
- vLLM's wheel metadata pins a stable torch (e.g. `torch==2.8.0`); pip refuses pre-release tags by default. Without this layer, `pip install vllm` would fail or silently downgrade.
- Same trick the NunchakuBased build uses (`Deployments/DockerContainers/Builds/Generative/Diffusion/NunchakuBased/Dockerfile.pytorch_reinstall`).

### nvidia-cudnn-frontend constraint stripped (`Dockerfile.pytorch_reinstall`)

- The NV container 25.06-py3's `/etc/pip/constraint.txt` pins `nvidia-cudnn-frontend==1.12.0`. vLLM 0.11.1+ pulls in `flashinfer-python==0.5.2` which requires `nvidia-cudnn-frontend>=1.13.0`. The constraint blocks vLLM install with `ResolutionImpossible`.
- Fix: same `sed -i '/^nvidia-cudnn-frontend==/d' /etc/pip/constraint.txt` pattern as the existing torch / numpy strips.
- Why we don't care about the NV pin: cudnn-frontend at 1.12.0 is a NV container assumption for *its* preinstalled DALI/RAPIDS stack, none of which is on MinerU's runtime path. flashinfer-python pulls 1.23.0 cleanly; runtime sanity check `import vllm` still passes.

### Flash-attn uninstalled after torch reinstall (`Dockerfile.pytorch_reinstall`)

- The NV container ships `flash-attn 2.7.4.post1` compiled against the alpha torch. After downgrading torch to stable, the prebuilt `flash_attn_2_cuda.so` fails to load with `undefined symbol: _ZN3c104cuda9SetDeviceEa` (a libtorch C++ ABI symbol). vLLM tries to import flash-attn during engine init and dies before reaching the model.
- Two paths considered: (a) recompile flash-attn against stable torch — 30-min nvcc build, requires preserving wheel cache; (b) uninstall and let vLLM fall back to xformers (already on the image). Picked (b). Throughput on long sequences drops ~10-20%; correctness unaffected. The Phase 1 P&ID workload is not throughput-bound.
- If flash-attn becomes important later (e.g. for batch inference on long context), reinstall it in `Dockerfile.pytorch_reinstall` with `pip install --no-build-isolation flash-attn` against the now-stable torch.

### Transformers pinned to 4.x (`Dockerfile.vllm` + `transformers_version` build_arg)

- vLLM 0.11.0 declares `transformers>=4.55.2` with no upper bound. transformers 5.x dropped legacy attributes like `PreTrainedTokenizerBase.all_special_tokens_extended` that vLLM 0.11.0 still calls in `get_cached_tokenizer`, so a fresh build resolves to transformers 5.x and crashes at `LLM(...)` load time with `AttributeError: Qwen2Tokenizer has no attribute all_special_tokens_extended`.
- Pin via `transformers_version` build arg, default `4.57.6` (latest 4.x). mineru-vl-utils' transformers extra also caps at `<5.0.0`, so this matches upstream intent.
- When bumping `vllm_version`: check vLLM's release notes for transformers 5.x compat. Once they ship support, this pin can be relaxed.
- **Forward-looking note:** vLLM 0.20.2 declares `transformers!=5.0.*,!=5.1.*,!=5.2.*,!=5.3.*,!=5.4.*,!=5.5.0,>=4.56.0` — so transformers 5.5.1+ becomes available once mineru-vl-utils lifts its `<0.12` vLLM cap. Don't try to skip ahead; vLLM 0.13–0.20.x has API breaks (engine init, logits-processor signatures) that would break MinerULogitsProcessor wiring.

### Pip dep-conflict warnings during build are expected

- After torch reinstall, pip prints warnings for `thinc`, `numba`, `cuml`, `dask-cuda`, `cugraph`, `dali`, `packaging`. These are pre-installed RAPIDS / spaCy / DALI packages that were pinned to older numpy/numba/packaging.
- **None of them are on MinerU's runtime path.** The build proceeds and the runtime sanity-checks (`import vllm`, `import mineru_vl_utils`) both pass.
- Do not "fix" by pinning the older numpy — vLLM and mineru-vl-utils need newer numpy.

### One image, not three

- The vLLM stack is the only nontrivial dep all three models share. Building one image keeps the iteration loop fast and avoids three separate ~36 GB layer caches.
- Phase 2 (Qwen3-VL) needs zero Docker changes — vLLM already supports `Qwen3VLForConditionalGeneration` on this image.
- Phase 3 (ColQwen2.5) likely needs zero Docker changes either; the work is upstream (LoRA merge against base model).

### Pydantic v2 `model_path` collision

- Pydantic v2 reserves the `model_` prefix for its own attributes. `MinerUConfiguration` uses `model_path` because that's the natural YAML key — and sets `protected_namespaces=()` in `ConfigDict` to silence the warning.
- If you add new pydantic models with `model_*` fields, do the same.

## Next steps per phase

### Phase 1 (MinerU) — finish

1. Run the smoke test above on the example PDF.
2. If extraction quality is poor on P&ID diagrams specifically, set `image_analysis: true` in `mineru_configuration.yml` — that turns on figure/chart analysis at extra latency.
3. Inspect a few `page_N.md` outputs vs the source PNGs in `<output_dir>/<pdf_stem>/` and decide whether the parsed tables match the part/instrumentation tables on each page.

### Phase 2 (Qwen3-VL-4B-Instruct via AWQ-8bit)

**Why AWQ-8bit not bf16:** Qwen3-VL-4B in bf16 (~8 GB weights) does not fit on a 12 GB Ampere GPU under vLLM 0.11.2 — the engine's `profile_run` at init unconditionally allocates a ~2.89 GiB vision-tower activation buffer on top of the loaded weights, exceeding the 11.63 GiB physical capacity. Tested at `max_model_len` 2048 / 4096 / 8192, all the same OOM. FP8 is hardware-accelerated only on sm_89+ (Ada/Hopper), so the 3060 (sm_86) can't use it either. **AWQ-8bit (`cyankiwi/Qwen3-VL-4B-Instruct-AWQ-8bit`)** is the chosen path: ~4 GB weights, runs on Ampere via vLLM's `quantization="compressed-tensors"` loader (cyankiwi published this through llm-compressor, which writes compressed-tensors format).

**Wrapper architecture:** Follows the QwenLM/Qwen3-VL upstream canonical inference pattern:

1. `transformers.AutoProcessor.from_pretrained(model_path)` — provides `apply_chat_template`
2. `qwen_vl_utils.process_vision_info(messages)` — Qwen-VL family image/video preprocessing (resize to model's expected pixel budget, etc.)
3. `vllm.LLM.generate([{"prompt": ..., "multi_modal_data": {"image": [...]}}], sampling_params=...)` — actual inference

The previous draft of the wrapper used `vllm.LLM.chat()`, which works for most VLMs but bypasses `process_vision_info`'s image budget logic; upstream specifically recommends the explicit path.

**Done (2026-05-10):**
- `Dockerfile.qwen3vl` build component installs `qwen-vl-utils==0.0.14` + `accelerate`. Added `qwen_vl_utils_version` build arg. Image needs rebuild to pick this up.
- `MoreMinerU/moremineru/Applications/Qwen3VLVLLM.py` refactored to the explicit `AutoProcessor` + `process_vision_info` + `llm.generate` path.
- `Qwen3VLConfiguration` default sampling params updated to the model card's recommended VL settings (`top_p=0.8`, `top_k=20`, `temperature=0.7`, `presence_penalty=1.5`).
- `smoke_qwen3vl_gpu.py` default model-path / quantization updated for AWQ-8bit; old bf16-OOM-fighting flags (`--max-num-seqs=1`, `--limit-images=1`) reverted to sane values.

**Done (2026-05-10, second session):**
- Image rebuilt with `Dockerfile.qwen3vl` layer (qwen-vl-utils 0.0.14 + accelerate 1.13.0 + av 17.0.1). New image sha `ea1221c0b9c5`.
- Smoke test passed against AWQ-8bit weights end-to-end (no manual `pip` patches). Per-page latency on RTX 3060: ~10-12 s.
- New `CLIPDFQwen3VLChat` app ran full corpus on `rev11.pdf` (7 pages, all `status: ok`, total ~76 s). Output is freeform `page_N.txt` + `manifest.json` (which now also records the prompt for reproducibility).
- Two memory-management gotchas surfaced (both now documented in `qwen3vl_configuration.yml.example`):
  - **`gpu_memory_utilization` must be ≥ 0.95** for `max_model_len=8192`: at 0.90, KV cache memory is 1.10 GiB but vLLM needs 1.12 GiB → ValueError at engine init (NOT an OOM, a sanity-check). At 0.95 we get 1.69 GiB KV cache.
  - **`image_max_pixels` / `image_min_pixels`** must be set on each image content item, else a 250-DPI rasterized P&ID page (~12 MP) produces ~11,466 visual tokens, overrunning `max_model_len=8192`. Defaults of `1280*28*28 = 1,003,520` (max) and `256*28*28 = 200,704` (min) cap to ~1,280 visual tokens, well under the limit. The wrapper attaches these from `Qwen3VLConfiguration` automatically.
- 13 unit tests pass (5 MinerU + 8 Qwen3VL, with 2 new tests for `image_max_pixels` / `image_min_pixels` defaults and disable-via-null behavior).

**Known limitation:** dense P&ID schematics produce some hallucinated component labels (e.g. "SCHNIPPERY" for what is probably "SOLENOID"). General-purpose VLMs aren't built for engineering OCR; MinerU is the right tool when you need verbatim labels. Use Qwen3-VL for descriptive / question-answering work where approximate identification suffices.

### Phase 3 (ColQwen2.5-v0.2)

What's on disk is only the LoRA adapter (240 MB, `peft` format, `base_model_name_or_path: vidore/colqwen2.5-base`). To use it:

1. Download `vidore/colqwen2.5-base` — ~7 GB. The user already has `colqwen2-v0.1-merged` from a prior round, suggesting they prefer to merge LoRA into base and ship a single merged dir rather than load the adapter at runtime.
2. Merge with peft:
   ```python
   from peft import PeftModel
   from transformers import AutoModel
   base = AutoModel.from_pretrained("/Data/Models/Multimodal/vidore/colqwen2.5-base", trust_remote_code=True)
   merged = PeftModel.from_pretrained(base, "/Data/Models/Multimodal/vidore/colqwen2.5-v0.2").merge_and_unload()
   merged.save_pretrained("/Data/Models/Multimodal/vidore/colqwen2.5-v0.2-merged")
   ```
3. Wrapper for retrieval workflow (different shape than MinerU/Qwen3-VL — ColQwen produces multi-vector embeddings for image patches, used with MaxSim against text query embeddings). Not a simple `LLM.generate()` call. See ColPali / ColQwen example notebooks.
4. Likely won't run cleanly via vLLM for the embedding side — ColQwen2.5 embedding extraction is typically done through `transformers` directly. The image has transformers available; the embedding code can sit alongside vLLM-served generation.

## File map

```
Deployments/DockerContainers/Builds/Multimodal/VLLMMultimodal/
  build_configuration.yml.example   # base image + pinned versions
  run_configuration.yml.example     # volume mounts (Data drive, repo, workspace)
  Dockerfile.system_deps            # poppler-utils + opencv libs
  Dockerfile.pytorch_reinstall      # alpha → stable torch swap
  Dockerfile.vllm                   # pinned vLLM + import sanity check
  Dockerfile.mineru                 # mineru-vl-utils[vllm] + pdf2image
  README.md                         # build/run commands
  STATUS.md                         # this file

PythonLibraries/HuggingFace/MoreMinerU/
  moremineru/
    Configurations/MinerUConfiguration.py    # pydantic, YAML load/save
    Applications/MinerU2_5ProVLLM.py         # wraps vllm.LLM + MinerUClient
  tests/test_mineru_configuration.py         # host-runnable, no vLLM needed

PythonApplications/CLIPDFExtraction/
  Configurations/
    mineru_configuration.yml.example
    pdf_extraction_configuration.yml.example
  Executables/main_CLIPDFExtraction.py
  clipdfextraction/
    ApplicationPaths.py
    CLIPDFExtraction.py
    Core/
      ProcessConfigurations.py
      PDFExtractionConfiguration.py
      PDFRasterizer.py                       # uses pdf2image (poppler under the hood)
      ExtractionRunner.py                    # per-PDF orchestration + manifest.json
```

## Build artifacts

- Current image: `vllm-multimodal:25.06-py3` (sha256:0df888508f1c...) — 39.5 GB. Built 2026-05-10 with all fixes baked in (transformers pin, flash-attn uninstall, cudnn-frontend constraint strip). Smoke test on rev01 PDF passed against this image's predecessor; this image needs a re-smoke-test to confirm clean-container path works without manual `pip` patches.
- Pre-fix image: dangling sha `8d80bfd4299f` (36.6 GB). Untagged after the rebuild. Safe to remove with `docker rmi 8d80bfd4299f` to reclaim disk; only kept around in case a hot-fix needs to be reproduced.
- Build args used: `torch_version=2.9.0`, `torchvision_version=0.24.0`, `numpy_version=2.1.2`, `vllm_version=0.11.2`, `transformers_version=4.57.6`, `mineru_vl_utils_version=0.2.7`
- All pins live in `build_configuration.yml`.

## How to bump versions later

The model card recommends `vllm-async-engine` for higher throughput (2.12 fps on A100). If the in-process LLM API hits throughput limits on the user's PDFs:

1. Bump `vllm_version` in `build_configuration.yml` (ceiling is `<0.12` per `mineru-vl-utils[vllm]` constraint as of 0.2.7; check `https://pypi.org/project/mineru-vl-utils/` for newer ceilings before bumping `mineru_vl_utils_version`).
2. Switch `MinerU2_5ProVLLM.load()` to construct `vllm.AsyncLLMEngine` instead of `LLM`, and update `extract_from_images` to use async submission.
3. Rebuild: `docker_builder build .` (the layer chain is small enough that only the vLLM + mineru layers reinstall).

## What NOT to do

- Don't move to NVIDIA PyTorch 25.07-py3 or later without a separate validation cycle (CUDA 13).
- Don't add `vllm` or `mineru-vl-utils` to `CommonComponents/` — they're specific to this image.
- Don't hardcode model paths in Python code — everything goes through YAML configs (project rule).
- Don't pin numpy back to 1.x to silence the build warnings — vLLM and mineru-vl-utils require numpy 2.x.
- Don't commit the actual `build_configuration.yml` / `run_configuration.yml` (they're user-customized copies of the `.example` files); only the `.example` files are committed.
