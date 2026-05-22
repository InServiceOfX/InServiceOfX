# AGENTS.md - CLIImage Config Studio

This project is the local GUI for configuring CLIImage YAML files before model
loading or image generation.

## Scope

- Frontend: Vite TypeScript in `frontend/`.
- Visual workflow overview: `litegraph.js`.
- Backend: Rust HTTP server in `backend/`.
- Backend owns local file IO. Prefer Rust over Python for new backend behavior.
- This app configures YAML. It should not load diffusion models itself.

## User Preferences

- Keep YAML as the source of truth for generation settings.
- Do not reintroduce command-line overrides for prompt, width, height, steps,
  guidance, batch size, or similar values.
- LoRA switching should be GUI-friendly toggles/strength controls that write
  `nunchaku_loras_configuration.yml`.
- Profiles are useful and should remain separate from tracked example configs.

## Run

From the repo root:

```bash
cargo run --manifest-path Typescript/CLIImageConfigStudio/backend/Cargo.toml -- --host 127.0.0.1 --port 8876
```

In another terminal:

```bash
cd Typescript/CLIImageConfigStudio/frontend
npm install
npm run dev -- --host 127.0.0.1 --port 5173
```

Open `http://127.0.0.1:5173/`.

## Before Editing

Read `HANDOFF.md` first. It records the branch state, verification already
done, ignored local files, and next priorities.

Do not commit generated files or local user configs:

- `backend/target/`
- `frontend/dist/`
- `frontend/node_modules/`
- `PythonApplications/CLIImage/Configurations/*.yml`
- `PythonApplications/CLIImage/Configurations/profiles/`
