from corecode.Utilities import (
    DataSubdirectories,
    )

from transformers import BitsAndBytesConfig as BitsAndBytesConfig, \
    T5EncoderModel
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig, \
    FluxTransformer2DModel, FluxPipeline

from morediffusers.Configurations import DiffusionPipelineConfiguration

from pathlib import Path

import torch

data_sub_dirs = DataSubdirectories()
test_data_directory = Path(__file__).resolve().parents[1] / "TestData"

def test_load_with_bitsandbytes_bfloat16():
    """
    Fails for 3060, CUDA Out of Memory
    """
    quant_config = BitsAndBytesConfig(load_in_8bit=True)

    test_file_path = test_data_directory / "flux_pipeline_configuration_empty.yml"
    configuration = DiffusionPipelineConfiguration(test_file_path)

    pretrained_diffusion_model_name_or_path = \
        data_sub_dirs.ModelsDiffusion / "black-forest-labs" / "FLUX.1-dev"
    if not pretrained_diffusion_model_name_or_path.exists():

        pretrained_diffusion_model_name_or_path = \
            Path("/Data1/Models/Diffusion/") / "black-forest-labs" / "FLUX.1-dev"

    configuration.diffusion_model_path = pretrained_diffusion_model_name_or_path
    configuration.torch_dtype = torch.bfloat16

    text_encoder_8bit = T5EncoderModel.from_pretrained(
        pretrained_diffusion_model_name_or_path,
        subfolder="text_encoder_2",
        quantization_config=quant_config,
        torch_dtype=configuration.torch_dtype)

    quant_config = DiffusersBitsAndBytesConfig(load_in_8bit=True)

    transformer_8bit = FluxTransformer2DModel.from_pretrained(
        pretrained_diffusion_model_name_or_path,
        subfolder="transformer",
        quantization_config=quant_config,
        torch_dtype=configuration.torch_dtype)


def test_load_with_bitsandbytes_float16():
    """
    Fails for 3060, CUDA Out of Memory
    """

    quant_config = BitsAndBytesConfig(load_in_8bit=True)

    test_file_path = test_data_directory / "flux_pipeline_configuration_empty.yml"
    configuration = DiffusionPipelineConfiguration(test_file_path)

    pretrained_diffusion_model_name_or_path = \
        data_sub_dirs.ModelsDiffusion / "black-forest-labs" / "FLUX.1-dev"
    if not pretrained_diffusion_model_name_or_path.exists():

        pretrained_diffusion_model_name_or_path = \
            Path("/Data1/Models/Diffusion/") / "black-forest-labs" / "FLUX.1-dev"

    configuration.diffusion_model_path = pretrained_diffusion_model_name_or_path
    configuration.torch_dtype = torch.float16

    text_encoder_8bit = T5EncoderModel.from_pretrained(
        pretrained_diffusion_model_name_or_path,
        subfolder="text_encoder_2",
        quantization_config=quant_config,
        torch_dtype=configuration.torch_dtype)

    quant_config = DiffusersBitsAndBytesConfig(load_in_8bit=True)

    transformer_8bit = FluxTransformer2DModel.from_pretrained(
        pretrained_diffusion_model_name_or_path,
        subfolder="transformer",
        quantization_config=quant_config,
        torch_dtype=configuration.torch_dtype)