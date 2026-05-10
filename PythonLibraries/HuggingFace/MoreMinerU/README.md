# MoreMinerU

Thin Python wrappers around [`mineru-vl-utils`](https://pypi.org/project/mineru-vl-utils/) for serving the `opendatalab/MinerU2.5-Pro-2604-1.2B` model with the vLLM in-process engine.

Designed to run inside the `VLLMMultimodal` Docker image (see `Deployments/DockerContainers/Builds/Multimodal/VLLMMultimodal/`). Configuration is YAML-driven via `MinerUConfiguration` — no hardcoded paths.

## Layout

```
moremineru/
  Configurations/
    MinerUConfiguration.py     # Pydantic config + YAML load/save
  Applications/
    MinerU2_5ProVLLM.py        # Wraps vllm.LLM + MinerUClient
```

## Use from Python

```python
from pathlib import Path
from PIL import Image
from moremineru.Configurations import MinerUConfiguration
from moremineru.Applications import MinerU2_5ProVLLM

config = MinerUConfiguration.from_yaml(Path("/path/to/mineru_configuration.yml"))
runner = MinerU2_5ProVLLM(config)
runner.load()
result = runner.extract_from_image(Image.open("/path/to/page.png"))
print(result)
runner.release()
```

The CLI in `PythonApplications/CLIPDFExtraction/` drives this end-to-end on a directory of PDFs.
