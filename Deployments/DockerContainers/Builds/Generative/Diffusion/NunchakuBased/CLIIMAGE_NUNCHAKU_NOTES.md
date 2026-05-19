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
python3 Scripts/QuickAliases/RunCLIImageNunchaku.py --gpu-id 1
```

That starts CLIImage directly. To open a shell instead:

```bash
python3 Scripts/QuickAliases/RunCLIImageNunchaku.py --gpu-id 1 --shell
```

Run a CLIImage command and exit:

```bash
python3 Scripts/QuickAliases/RunCLIImageNunchaku.py \
  --gpu-id 1 \
  --command '.status'
```

Multiple commands run in order:

```bash
python3 Scripts/QuickAliases/RunCLIImageNunchaku.py \
  --gpu-id 1 \
  --command '.enable_lora "hero-v2.1"' \
  --command '.set_lora_strength "hero-v2.1" 0.9' \
  --command '.generate_image'
```

Equivalent generic command:

```bash
python3 Scripts/QuickAliases/QuickDockerBuilder.py run \
  Generative/Diffusion/NunchakuBased \
  --gpu-id 1 \
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

## Config Profiles

Live config files stay ignored, but you can save/apply named local profiles
under `PythonApplications/CLIImage/Configurations/profiles/`:

```bash
python3 Scripts/QuickAliases/CLIImageProfiles.py save portrait-test
python3 Scripts/QuickAliases/CLIImageProfiles.py list
python3 Scripts/QuickAliases/CLIImageProfiles.py show portrait-test
python3 Scripts/QuickAliases/CLIImageProfiles.py apply portrait-test
```

`apply` backs up the current live config into
`PythonApplications/CLIImage/Configurations/profiles/_backups/<timestamp>/`
before overwriting it. Profiles are intentionally gitignored because they can
contain private prompts, local model paths, or adult/private workflow variants.

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

CLIImage now has dot commands that toggle `is_active` by nickname and write the
same YAML file back:

```text
.status
.list_loras
.active_loras
.enable_lora "hero-v2.1"
.disable_lora "hero-v2.1"
.toggle_lora "hero-v2.1"
.set_lora_strength "hero-v2.1" 0.9
```

A later frontend can call these backend operations instead of editing comments.

## Highest Priority Improvements

1. Build a thin local UI now that backend commands exist. A LiteGraph UI
   could map nodes to existing YAML responsibilities: model, prompts,
   generation settings, LoRAs, control image, and batch output.
2. Add command-line prompt and output overrides so one-off generations do not
   require editing `pipeline_inputs.yml` and `flux_generation_configuration.yml`.

## Verification Notes

- YAML syntax for the six copied live CLIImage configs was parsed with
  `yaml.safe_load`.
- Local host Python does not currently have `pydantic`, so full model-level
  validation should be run inside the Nunchaku container or a project virtual
  environment.
- `nvidia-smi` was not available from the current host command environment, so
  the shortcut defaults to `--gpu-id 1` but accepts another GPU id.
