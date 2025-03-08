from corecode.Utilities import (
    DataSubdirectories,
    )

from pathlib import Path

import torch
from diffusers import StableDiffusion3Pipeline

data_sub_dirs = DataSubdirectories()
test_data_directory = Path(__file__).resolve().parents[1] / "TestData"

pretrained_diffusion_model_name_or_path = \
    data_sub_dirs.ModelsDiffusion / "tensorart" / "stable-diffusion-3.5-medium-turbo"
if not pretrained_diffusion_model_name_or_path.exists():

    pretrained_diffusion_model_name_or_path = \
        Path("/Data1/Models/Diffusion/") / "tensorart" / "stable-diffusion-3.5-medium-turbo"

def test_sd35mediumturbo_and_StableDiffusion3Pipeline_from_pretrained_works():
    pipe = StableDiffusion3Pipeline.from_pretrained(
        pretrained_diffusion_model_name_or_path,
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
        local_files_only=True
    )
    pipe = pipe.to("cuda")

from optimum.quanto import freeze, qint4, quantize, quantization_map

from diffusers.models.transformers import SD3Transformer2DModel

def test_sd35mediumturbo_quantized_transformer():
    transformer = SD3Transformer2DModel.from_pretrained(
        pretrained_diffusion_model_name_or_path,
        subfolder="transformer",
        torch_dtype=torch.bfloat16
        )
    quantize(transformer, weights=qint4)
    freeze(transformer)

    transformer.save_pretrained("sd35mediumturbo-transformer-bfloat16-qint4")
    qmap = quantization_map(transformer)
    with open("quanto_qmap.json", "w", encoding="utf8") as f:
        json.dump(qmap, f, indent=4)
