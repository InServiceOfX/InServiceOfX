# CLIImage Nunchaku Notes

## Current State

- Use `Scripts/QuickAliases/QuickDockerBuilder.py`; the older
  `Scripts/QuickAliases/QuickRunDocker.py` is not wired to this deployment.
- This deployment now has local ignored copies of `build_configuration.yml` and
  `run_configuration.yml`, so `QuickDockerBuilder.py list` reports
  `Generative/Diffusion/NunchakuBased` as ready.
- `run_configuration.yml` mounts this workspace checkout at `/InServiceOfX`.
  That keeps host edits in
  `/home/propdev/.openclaw/workspace/workspace2/repos/InServiceOfX` visible in
  the container.
- CLIImage live configs were copied from the old working tree into
  `PythonApplications/CLIImage/Configurations/*.yml`. They are intentionally
  gitignored; keep tracked defaults in `*.yml.example`.

## Launch

Preferred shortcut:

```bash
python3 Scripts/QuickAliases/RunCLIImageNunchaku.py --gpu-id 0
```

That starts CLIImage directly. To open a shell instead:

```bash
python3 Scripts/QuickAliases/RunCLIImageNunchaku.py --gpu-id 0 --shell
```

Equivalent generic command:

```bash
python3 Scripts/QuickAliases/QuickDockerBuilder.py run \
  Generative/Diffusion/NunchakuBased \
  --gpu-id 0 \
  --entrypoint /bin/bash \
  -- -lc 'cd /InServiceOfX/PythonApplications/CLIImage && python3 Executables/main_CLIImage.py --dev'
```

If launched with `--shell`, run this inside the container:

```bash
cd /InServiceOfX/PythonApplications/CLIImage
python3 Executables/main_CLIImage.py --dev
```

`--dev` points CLIImage at
`/InServiceOfX/PythonApplications/CLIImage/Configurations`.

## Configuration Map

| CLIImage YAML | Python model |
| --- | --- |
| `batch_processing_configuration.yml` | `morediffusers.Configurations.BatchProcessingConfiguration` |
| `flux_generation_configuration.yml` | `morediffusers.Configurations.FluxGenerationConfiguration` |
| `nunchaku_configuration.yml` | `morediffusers.Configurations.NunchakuConfiguration` |
| `nunchaku_flux_control_configuration.yml` | `morediffusers.Configurations.NunchakuFluxControlConfiguration` |
| `nunchaku_loras_configuration.yml` | `morediffusers.Configurations.NunchakuLoRAsConfiguration` |
| `pipeline_inputs.yml` | `morediffusers.Configurations.PipelineInputs` |

`PythonApplications/CLIImage/cliimage/Core/ProcessConfigurations.py` loads
those files and passes them into `FluxNunchakuAndLoRAs`,
`FluxDepthNunchakuAndLoRAs`, and `FluxKontextNunchakuAndLoRAs`.

## LoRA Switching

The current LoRA YAML schema already supports switching without commenting:

```yaml
loras:
  - nickname: some_lora
    directory_path: /Data/...
    filename: model.safetensors
    lora_strength: 0.8
    is_active: true
```

`NunchakuLoRAsConfiguration.get_valid_loras()` ignores entries where
`is_active` is false. The copied working config currently has 42 LoRA entries,
with 3 active and 39 inactive.

The next low-risk improvement is a small CLI command that toggles
`is_active` by nickname and writes the YAML back. A later frontend can call the
same backend operation instead of editing comments.

## Highest Priority Improvements

1. Add a non-interactive CLIImage command path, for example:
   `main_CLIImage.py --dev --command .generate_image`.
   The Docker wrapper can now start CLIImage directly, but CLIImage still needs
   an interactive command prompt for generation.
2. Add LoRA management commands:
   `list_loras`, `enable_lora <nickname>`, `disable_lora <nickname>`,
   `set_lora_strength <nickname> <value>`.
3. Create named config profiles under an ignored directory such as
   `PythonApplications/CLIImage/Configurations/profiles/`, then add a tracked
   helper to copy or select profiles.
4. Build a thin local UI only after the backend commands exist. A LiteGraph UI
   could map nodes to existing YAML responsibilities: model, prompts,
   generation settings, LoRAs, control image, and batch output.

## Verification Notes

- YAML syntax for the six copied live CLIImage configs was parsed with
  `yaml.safe_load`.
- Local host Python does not currently have `pydantic`, so full model-level
  validation should be run inside the Nunchaku container or a project virtual
  environment.
- `nvidia-smi` was not available from the current host command environment, so
  the shortcut defaults to `--gpu-id 0` but accepts another GPU id.
