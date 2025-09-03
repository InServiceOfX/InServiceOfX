from corecode.Utilities import DataSubdirectories, is_model_there

from morediffusers.Applications import FluxDepthNunchakuAndLoRAs
from morediffusers.Configurations import (
    FluxGenerationConfiguration,
    PipelineInputs,
    NunchakuFluxControlConfiguration,
    NunchakuLoRAsConfiguration)

import pytest

data_subdirectories = DataSubdirectories()

relative_model_path = "Models/Diffusion/black-forest-labs/FLUX.1-dev"

is_model_downloaded, model_path = is_model_there(
    relative_model_path,
    data_subdirectories)

relative_nunchaku_flux_model_path = "Models/Diffusion/nunchaku-tech/nunchaku-flux.1-depth-dev"

is_nunchaku_flux_model_downloaded, nunchaku_flux_model_path = is_model_there(
    relative_nunchaku_flux_model_path,
    data_subdirectories)

relative_nunchaku_t5_model_path = "Models/Diffusion/nunchaku-tech/nunchaku-t5"

is_nunchaku_t5_model_downloaded, nunchaku_t5_model_path = is_model_there(
    relative_nunchaku_t5_model_path,
    data_subdirectories)

relative_depth_model_path = "Models/Generation/LiheYoung/depth-anything-large-hf"

is_depth_model_downloaded, depth_model_path = is_model_there(
    relative_depth_model_path,
    data_subdirectories)

@pytest.mark.skipif(
    not is_model_downloaded or \
        not is_nunchaku_flux_model_downloaded or \
        not is_nunchaku_t5_model_downloaded or \
        not is_depth_model_downloaded,
    reason="Models not downloaded")
def test_FluxDepthNunchakuAndLoRAs_creates_prompt_embeds():

