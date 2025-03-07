from corecode.Utilities import (
    DataSubdirectories,
    )

from transformers import BitsAndBytesConfig as BitsAndBytesConfig, \
    T5EncoderModel
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig, \
    FluxTransformer2DModel

from morediffusers.Configurations import DiffusionPipelineConfiguration

from pathlib import Path

import torch

data_sub_dirs = DataSubdirectories()
test_data_directory = Path(__file__).resolve().parents[1] / "TestData"

pretrained_diffusion_model_name_or_path = \
    data_sub_dirs.ModelsDiffusion / "black-forest-labs" / "FLUX.1-dev"
if not pretrained_diffusion_model_name_or_path.exists():

    pretrained_diffusion_model_name_or_path = \
        Path("/Data1/Models/Diffusion/") / "black-forest-labs" / "FLUX.1-dev"

pretrained_schnell_model_path = \
    data_sub_dirs.ModelsDiffusion / "black-forest-labs" / "FLUX.1-schnell"
if not pretrained_schnell_model_path.exists():

    pretrained_schnell_model_path = \
        Path("/Data1/Models/Diffusion/") / "black-forest-labs" / "FLUX.1-schnell"

def test_load_with_bitsandbytes_bfloat16():
    """
    Fails for 3060, CUDA Out of Memory
    """
    quant_config = BitsAndBytesConfig(load_in_8bit=True)

    test_file_path = test_data_directory / "flux_pipeline_configuration_empty.yml"
    configuration = DiffusionPipelineConfiguration(test_file_path)

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


def test_load_with_bitsandbytes_4bit():
    from transformers import BitsAndBytesConfig \
        as TransformersBitsAndBytesConfig
    quant_config = TransformersBitsAndBytesConfig(load_in_4bit=True)

    test_file_path = test_data_directory / "flux_pipeline_configuration_empty.yml"
    configuration = DiffusionPipelineConfiguration(test_file_path)

    configuration.diffusion_model_path = pretrained_diffusion_model_name_or_path
    configuration.torch_dtype = torch.bfloat16

    text_encoder_2_4bit = T5EncoderModel.from_pretrained(
        pretrained_diffusion_model_name_or_path,
        subfolder="text_encoder_2",
        quantization_config=quant_config,
        torch_dtype=configuration.torch_dtype)

    quant_config = DiffusersBitsAndBytesConfig(load_in_4bit=True)

    transformer_4bit = FluxTransformer2DModel.from_pretrained(
        pretrained_diffusion_model_name_or_path,
        subfolder="transformer",
        quantization_config=quant_config,
        torch_dtype=configuration.torch_dtype)

def test_load_with_bitsandbytes_4bit_schnell():
    """
    E       torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 90.00 MiB. GPU 0 has a total capacity of 11.75 GiB of which 81.94 MiB is free. Process 46840 has 11.66 GiB memory in use. Of the allocated memory 11.37 GiB is allocated by PyTorch, and 174.05 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)

    /usr/local/lib/python3.10/dist-packages/bitsandbytes/nn/modules.py:295: OutOfMemoryError
    """
    from transformers import BitsAndBytesConfig \
        as TransformersBitsAndBytesConfig
    quant_config = TransformersBitsAndBytesConfig(load_in_4bit=True)

    test_file_path = test_data_directory / "flux_pipeline_configuration_empty.yml"
    configuration = DiffusionPipelineConfiguration(test_file_path)

    configuration.diffusion_model_path = pretrained_schnell_model_path
    configuration.torch_dtype = torch.bfloat16

    text_encoder_2_4bit = T5EncoderModel.from_pretrained(
        pretrained_schnell_model_path,
        subfolder="text_encoder_2",
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16)

    quant_config = DiffusersBitsAndBytesConfig(load_in_4bit=True)

    transformer_4bit = FluxTransformer2DModel.from_pretrained(
        pretrained_diffusion_model_name_or_path,
        subfolder="transformer",
        quantization_config=quant_config,
        torch_dtype=configuration.torch_dtype)