from corecode.Utilities import (
    DataSubdirectories,
    )

from pathlib import Path

import torch
from diffusers import StableDiffusion3Pipeline

data_sub_dirs = DataSubdirectories()
test_data_directory = Path(__file__).resolve().parents[1] / "TestData"

pretrained_diffusion_model_name_or_path = \
    data_sub_dirs.ModelsDiffusion / "stabilityai" / "stable-diffusion-3-medium"
if not pretrained_diffusion_model_name_or_path.exists():

    pretrained_diffusion_model_name_or_path = \
        Path("/Data1/Models/Diffusion/") / "stabilityai" / "stable-diffusion-3-medium"

def test_sd3medium_and_StableDiffusion3Pipeline_from_pretrained():
    """
    Fails on 3060 with 12GB VRAM:

    E           torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 80.00 MiB. GPU 0 has a total capacity of 11.75 GiB of which 62.94 MiB is free. Process 33234 has 11.68 GiB memory in use. Of the allocated memory 11.37 GiB is allocated by PyTorch, and 211.54 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)

    /usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py:1329: OutOfMemoryError
    """
    pipe = StableDiffusion3Pipeline.from_pretrained(
        pretrained_diffusion_model_name_or_path,
        torch_dtype=torch.float16,
    )
    pipe = pipe.to("cuda:0")

