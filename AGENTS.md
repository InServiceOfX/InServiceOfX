# AGENTS.md — InServiceOfX entry point for AI agents

You are an AI coding agent (Claude Code, Codex, OpenClaw, etc.) opened against this repo. This file is the **top-level index** for agent work — read this first, then descend into the specific subarea you're asked to advance.

This is *not* the human onboarding doc — see `README.md` for that.

## What this repo is

InServiceOfX is the user's deep-learning monorepo. It bundles:

- **Python libraries** (`PythonLibraries/`) — reusable, framework-specific wrappers around HuggingFace / vLLM / etc.
- **Python applications** (`PythonApplications/`) — single-purpose CLI tools that compose those libraries to solve a concrete task.
- **Docker deployments** (`Deployments/DockerContainers/Builds/`) — pinned, reproducible images that bundle the system / CUDA / Python deps needed for each workload.
- **Build orchestration in Rust** (`RustLibraries/docker_builder/`) — reads each deployment's `build_configuration.yml` and `run_configuration.yml` and shells out to `docker build` / `docker run` with the right args.
- **Scripts** (`Scripts/`) — quality-of-life helpers, including `QuickAliases/QuickDockerBuilder.py` which wraps the Rust binary so agents don't have to remember its target/debug path.

Each Docker deployment is independent — they don't share an image. They DO often share component Dockerfile pieces under `Deployments/DockerContainers/CommonComponents/`.

## Active areas with their own AGENTS files

If the user points you at one of these, read its `AGENTS.md` (which is more concrete than this file):

| Area | Path | Status |
|---|---|---|
| **Multimodal vLLM image** (MinerU2.5-Pro + Qwen3-VL-4B AWQ-8bit + ColQwen2.5-v0.2) | `Deployments/DockerContainers/Builds/Multimodal/VLLMMultimodal/AGENTS.md` | Phases 1-3 smoke-tested 2026-05-10; Phase 4 (retrieval CLIs) open — see its `NEXT_STEPS.md` |
| **CuLLM FlashAttention-from-first-principles** (CUDA, forward+backward, causal, multi-head via cuBLASLt) | `CUDALibraries/CuLLM/AGENTS.md` | Forward MHA (projections + FlashAttention core) complete and end-to-end tested as of 2026-07-02; backward pass exists single-head only, no multi-head wiring yet — see its "What's NOT done" |

If you're asked about something not in this table, check:
- `Deployments/DockerContainers/Builds/*/STATUS.md` for build-specific status
- Recent git log on the relevant branch
- The user (when in doubt — see "When to stop and ask" below)

## Repo-wide conventions

### Git policy (CRITICAL — the user is explicit about these)

- **NEVER** commit or push to `master` or `main`. The user merges manually. Create / use feature branches.
- Branch naming: `feat/<short-name>` for features, `fix/<short-name>` for fixes. Some legacy uses other prefixes (`lab/`, etc.) — match what's already on the branch you're touching.
- Don't squash or rewrite history without explicit user request.
- Pushes to origin on feature branches are fine.

### Python environment

- The host uses `uv` for Python environments. Never use system `pip` or conda.
- Inside Docker containers, `pip` IS the right tool (the image is its own isolated environment).
- Python 3.12 in the multimodal image (NV PyTorch container 25.06-py3); Python 3.13 in the user's host uv envs.

### Configuration files

Every Docker deployment and every CLI app follows the same shape:

```
SomeArea/
  Configurations/
    foo_configuration.yml.example   # tracked in git, recommended defaults
    foo_configuration.yml           # gitignored, user-customized
```

`.gitignore` has explicit lines for each app's `Configurations/*.yml` plus globally `**/build_configuration.yml` and `**/run_configuration.yml`. **NEVER `git add` a live config** — start from the `.example` copy.

### C++ style (for the heterosplat workstream — not multimodal)

`snake_case` for members and functions. Spell out names — no `numel` / `nbytes` / other C-style abbreviations. (Documented in the user's auto-memory.)

### Where outputs land

- Model weights: `/media/propdev/9dc1a908-7eff-4e1c-8231-ext4/home/propdev/Data/Models/...` on the host (mounted at `/Data/` inside containers).
- HF cache: `/media/propdev/.../Data/.cache/huggingface/` on host (mounted at `/root/.cache/huggingface/` inside containers; survives `docker run --rm` only because the host dir is on the data drive).
- Generated outputs: `/home/propdev/.openclaw/workspace/workspace2/Data/Generated/...` (mounted at `/Workspace/Generated/...`).

### Build / run pattern

Always use the `QuickDockerBuilder.py` wrapper rather than invoking `docker build` / `docker run` directly. It:

- Auto-resolves the `docker_builder` Rust binary by walking up from its own location to the repo root.
- Auto-runs `cargo build` if the binary doesn't exist yet (Rust toolchain required).
- Lets you pass a deployment **short name** (e.g. `Multimodal/VLLMMultimodal`) instead of the full path.

```bash
# Build:
python3 <repo>/Scripts/QuickAliases/QuickDockerBuilder.py build Multimodal/VLLMMultimodal

# Run interactively:
python3 <repo>/Scripts/QuickAliases/QuickDockerBuilder.py run Multimodal/VLLMMultimodal --gpu-id 1 --entrypoint /bin/bash

# List available deployments:
python3 <repo>/Scripts/QuickAliases/QuickDockerBuilder.py list
```

### Dev loop inside containers

Every deployment's `run_configuration.yml` mounts the InServiceOfX repo at `/InServiceOfX` inside the container. **Edit Python on the host, re-run inside the container** — no rebuild needed unless `Dockerfile.*` / `build_configuration.yml` changes.

The Rust binary itself does need rebuilding from time to time (`cargo build` in `RustLibraries/docker_builder/`); `QuickDockerBuilder.py` handles that automatically on first invocation.

## When to stop and ask the user

- Risky / hard-to-reverse actions: `docker build --no-cache` (full rebuild, ~20 min), large model downloads (anything new and ≥ 5 GB), force-pushes, branch deletions.
- Anything that touches a GPU device other than the one the user has explicitly authorized for the current task.
- Conflicting version pins where you have to pick winners between two of the existing pinned libraries — read the relevant Decisions section in `STATUS.md` first, then ask.
- Three failed attempts at the same smoke test — capture the actual error trace in the relevant `STATUS.md` and stop. Don't iterate blindly.

## Memory and discovery

If you're a Claude Code session, the user has a persistent auto-memory at `/home/propdev/.claude/projects/-home-propdev--openclaw-workspace/memory/`. The `MEMORY.md` index there points at project-specific memories. The InServiceOfX entries you'll most often want are:

- `project_inserviceofx_vllm_multimodal.md` — multimodal stack state.
- `feedback_naming_conventions.md` — C++ style for adjacent work.

Codex / OpenClaw / other agents without that memory should rely on the in-repo `STATUS.md` + `AGENTS.md` files for context; they are designed to be self-sufficient.
