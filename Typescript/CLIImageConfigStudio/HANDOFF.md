# CLIImage Config Studio Handoff

Last updated: 2026-05-22.

## Branch

Current branch:

```text
feat/cliimage-command-and-lora-tools
```

Repo root:

```text
/home/propdev/.openclaw/workspace/workspace2/repos/InServiceOfX
```

Recent commits on this branch:

```text
bc8b36f feat: add CLIImage config studio
78eda1e feat: apply CLIImage profiles from launcher
b9c6b4b Revert "feat: add CLIImage run overrides"
992bd8d feat: add CLIImage status command
cb0e343 feat: add CLIImage config profiles
a688f92 feat: add CLIImage command and LoRA tools
```

The command-line override idea was intentionally reverted. The agreed model is:

- YAML/profiles define settings.
- The GUI edits YAML before model loading/generation.
- CLI commands are for actions/status and backend primitives.

Do not commit or push to `master`; Ernest merges feature branches manually.

## What Exists

`Typescript/CLIImageConfigStudio/` contains the YAML configuration GUI:

- `backend/`: Rust HTTP backend for local file IO.
- `frontend/`: Vite/TypeScript frontend using `litegraph.js`.
- `AGENTS.md`: project-local instructions for Codex, Claude Code, and
  OpenClaw sessions started inside this subdirectory.
- `README.md`: run instructions.

The GUI edits:

- `pipeline_inputs.yml`
- `flux_generation_configuration.yml`
- `batch_processing_configuration.yml`
- `nunchaku_loras_configuration.yml`

It reads status from:

- `nunchaku_configuration.yml`
- `nunchaku_flux_control_configuration.yml`

It can switch to another configuration directory and initialize missing live
YAML files from the tracked CLIImage `.yml.example` templates.

Related committed tooling:

- `Scripts/QuickAliases/RunCLIImageNunchaku.py`: starts the Nunchaku Docker
  CLIImage environment, supports `--profile`, `--command`, and `--shell`.
- `Scripts/QuickAliases/CLIImageProfiles.py`: saves/applies ignored local
  profiles under `PythonApplications/CLIImage/Configurations/profiles/`.
- `PythonApplications/CLIImage/Executables/main_CLIImage.py`: supports
  `--command` for non-interactive dot commands.
- `PythonLibraries/HuggingFace/MoreDiffusers/morediffusers/Configurations/NunchakuLoRAsConfiguration.py`:
  saves LoRA YAML and serializes paths cleanly.

The current LoRA switching model uses `is_active` and `lora_strength` in
`nunchaku_loras_configuration.yml`. The GUI should keep editing those YAML
fields instead of reintroducing command-line generation overrides.

## How To Run

From repo root:

```bash
cargo run --manifest-path Typescript/CLIImageConfigStudio/backend/Cargo.toml -- --host 127.0.0.1 --port 8876
```

In a second terminal:

```bash
cd Typescript/CLIImageConfigStudio/frontend
npm install
npm run dev -- --host 127.0.0.1 --port 5173
```

Open:

```text
http://127.0.0.1:5173/
```

If `8876` is already in use, it is probably a prior backend session. Stop it
with Ctrl+C if possible, or run the backend on another port and update
`frontend/vite.config.ts`.

Quick port check:

```bash
ss -ltnp | grep ':8876'
```

Direct CLIImage Docker smoke without opening the GUI:

```bash
python3 Scripts/QuickAliases/RunCLIImageNunchaku.py --gpu-id 1 --command .status
```

## Verification Already Done

These passed:

```bash
cargo check --manifest-path Typescript/CLIImageConfigStudio/backend/Cargo.toml
cargo fmt --manifest-path Typescript/CLIImageConfigStudio/backend/Cargo.toml --check
cd Typescript/CLIImageConfigStudio/frontend && npm run build
cd Typescript/CLIImageConfigStudio/frontend && npm audit --omit=dev --json
```

Backend smoke tests were also done manually against:

- `/health`
- `/api/status`
- `/api/location` with `initialize_from_examples: true`
- `/api/config`

`npm run build` warns that `litegraph.js` uses `eval` internally and that the
bundle is larger than 500 kB. This is from `litegraph.js`; acceptable for the
local tool for now.

Docker-based CLIImage integration tests passed earlier in this branch for:

- `PythonApplications/CLIImage/tests/integration_tests/Core/test_NunchakuLoRAsConfiguration.py`
- `PythonApplications/CLIImage/tests/integration_tests/Terminal/test_CommandHandler.py`
- `PythonApplications/CLIImage/tests/integration_tests/Scripts/test_CLIImageProfiles.py`

Not yet verified after the Config Studio commit:

- A full `.generate_image` run through the Nunchaku Docker image.
- Automated Rust backend tests.
- Automated frontend/browser tests.

## Local Ignored State

Expected ignored files/directories:

- `Typescript/CLIImageConfigStudio/backend/target/`
- `Typescript/CLIImageConfigStudio/frontend/node_modules/`
- `Typescript/CLIImageConfigStudio/frontend/dist/`
- `PythonApplications/CLIImage/Configurations/*.yml`
- `PythonApplications/CLIImage/Configurations/profiles/`
- `Deployments/DockerContainers/Builds/Generative/Diffusion/NunchakuBased/build_configuration.yml`
- `Deployments/DockerContainers/Builds/Generative/Diffusion/NunchakuBased/run_configuration.yml`

A local ignored profile named `codex-current-smoke` may exist from testing.

`Typescript/CLIImageConfigStudio/backend/Cargo.lock` is currently ignored by
the repo-wide `Cargo.lock` rule. Since this backend is an executable app, a
future cleanup can decide whether to force-add it or adjust the ignore rules.

## Related CLI Tools

Nunchaku Docker launcher:

```bash
python3 Scripts/QuickAliases/RunCLIImageNunchaku.py --command .status
```

Profile helper:

```bash
python3 Scripts/QuickAliases/CLIImageProfiles.py list
python3 Scripts/QuickAliases/CLIImageProfiles.py save <name>
python3 Scripts/QuickAliases/CLIImageProfiles.py apply <name>
```

## Hardware Constraint — RTX 3070 VRAM

The target GPU is an RTX 3070 (~8 GB VRAM). Generation is VRAM-intensive: even
the browser and cursor compositor consume measurable VRAM during a run. For
that reason, **the GUI does not trigger image generation**. The GUI's job is
purely YAML editing and profile management. Generation stays in the CLIImage
Docker terminal session where VRAM can be managed explicitly (close browser,
stop compositor, etc.).

`.status` is acceptable from the GUI because it is a lightweight read-only
query. `.generate_image` is not.

## Next Priorities

1. Add a `.status` shortcut in the GUI — either a Rust backend endpoint that
   shells out to `RunCLIImageNunchaku.py --command .status` and streams the
   result, or a "Copy `.status` Command" button (already present on the
   Overview panel). Do **not** add a `.generate_image` button.
2. Improve the LiteGraph canvas: add `onDrawForeground` callbacks so each node
   shows its live value on the canvas face (prompt preview, dimensions like
   "832×1216", active LoRA count). Color-code nodes by type (models=blue,
   prompts=green, loras=amber, generation=purple, output=gray). Double-click a
   node to jump to its corresponding sidebar tab. Keep the canvas read-only
   (no execution wires, no subgraphs, no minimap) to stay lightweight.
3. Improve the Location tab UX: add validation and clearer feedback for new
   config directories initialized from `.yml.example`.
4. Add Rust backend tests around config load/save, profile apply, and
   initialize-from-examples.
5. Add a minimal browser smoke test for the frontend.
6. Clean up DockerBuilder run-time warnings about missing build-only Dockerfile
   components.
