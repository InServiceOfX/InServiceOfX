"""
See
https://github.com/mit-han-lab/nunchaku/blob/main/examples/flux.1-depth-dev-lora.py
"""

from corecode.Utilities import (
    clear_torch_cache_and_collect_garbage,
    DataSubdirectories,
    )

from diffusers import FluxControlPipeline

from nunchaku import NunchakuFluxTransformer2dModel

from pathlib import Path
from time import sleep

data_sub_dirs = DataSubdirectories()

def test_with_nunchaku_flux_depth_repository():

    path = data_sub_dirs.ModelsDiffusion / "mit-han-lab" / \
        "nunchaku-flux.1-depth-dev" / \
        "svdq-int4_r32-flux.1-depth-dev.safetensors"

    if not path.exists():
        path = Path("/Data1/Models/Diffusion/") / "mit-han-lab" / \
            "nunchaku-flux.1-depth-dev" / \
            "svdq-int4_r32-flux.1-depth-dev.safetensors"

    print("path", path)
    print("path.exists()", path.exists())

    # TODO: This freezes.
    #transformer = NunchakuFluxTransformer2dModel.from_pretrained(str(path))

def test_with_svdq_int4_flux_depth_repository():

    path = data_sub_dirs.ModelsDiffusion / "mit-han-lab" / \
        "svdq-int4-flux.1-depth-dev"

    if not path.exists():
        path = Path("/Data1/Models/Diffusion/") / "mit-han-lab" / \
            "svdq-int4-flux.1-depth-dev"

    print("path", path)
    print("path.exists()", path.exists())

    transformer = NunchakuFluxTransformer2dModel(str(path))

