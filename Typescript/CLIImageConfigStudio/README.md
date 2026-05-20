# CLIImage Config Studio

TypeScript/Rust GUI for editing CLIImage YAML files before model loading and
generation.

The frontend is a Vite TypeScript app with `litegraph.js` for a visual workflow
overview. The backend is Rust and owns all local file IO.

## Run

From the repository root, start the Rust backend:

```bash
cargo run --manifest-path Typescript/CLIImageConfigStudio/backend/Cargo.toml
```

In another terminal:

```bash
cd Typescript/CLIImageConfigStudio/frontend
npm install
npm run dev
```

Open the Vite URL, usually `http://127.0.0.1:5173`.

## Scope

This app edits local YAML only. It does not load diffusion models and does not
generate images.

Editable YAML:

- `pipeline_inputs.yml`
- `flux_generation_configuration.yml`
- `batch_processing_configuration.yml`
- `nunchaku_loras_configuration.yml`

Read-only status also summarizes:

- `nunchaku_configuration.yml`
- `nunchaku_flux_control_configuration.yml`

Profiles are saved under the ignored directory:

```text
PythonApplications/CLIImage/Configurations/profiles/
```

The Location tab can point the GUI at another configuration directory. If the
directory is new, enable "Initialize missing YAML from .example templates" and
the Rust backend will create the six live `.yml` files from the tracked
CLIImage `.yml.example` files before switching to that directory.
