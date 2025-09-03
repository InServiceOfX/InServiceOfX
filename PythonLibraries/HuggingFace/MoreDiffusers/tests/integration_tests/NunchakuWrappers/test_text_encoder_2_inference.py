from corecode.Utilities import DataSubdirectories, is_model_there
from morediffusers.Configurations import (
    NunchakuConfiguration)

from morediffusers.NunchakuWrappers import (
    text_encoder_2_inference,
    )

from morediffusers.Wrappers.pipelines import (
    change_pipe_to_cuda_or_not,
)

import pytest
import torch

data_subdirectories = DataSubdirectories()

relative_model_path = "Models/Diffusion/black-forest-labs/FLUX.1-dev"

is_model_downloaded, model_path = is_model_there(
    relative_model_path,
    data_subdirectories)

relative_nunchaku_model_path = "Models/Diffusion/jib-mix-svdq"

is_nunchaku_model_downloaded, nunchaku_model_path = is_model_there(
    relative_nunchaku_model_path,
    data_subdirectories)

relative_nunchaku_t5_model_path = "Models/Diffusion/nunchaku-tech/nunchaku-t5"

is_nunchaku_t5_model_downloaded, nunchaku_t5_model_path = is_model_there(
    relative_nunchaku_t5_model_path,
    data_subdirectories)

@pytest.mark.skipif(
    not is_model_downloaded or \
        not is_nunchaku_model_downloaded or \
        not is_nunchaku_t5_model_downloaded,
    reason="Models not downloaded")
def test_create_flux_text_encoder_2_pipeline():
    nunchaku_t5_model_file_path = nunchaku_t5_model_path / \
        "awq-int4-flux.1-t5xxl.safetensors"

    configuration = NunchakuConfiguration(
        flux_model_path=model_path,
        nunchaku_model_path=nunchaku_model_path,
        nunchaku_t5_model_path=nunchaku_t5_model_file_path,
    )
    configuration.cuda_device = "cuda:0"
    configuration.torch_dtype = "bfloat16"

    assert configuration.torch_dtype == torch.bfloat16    

    text_encoder_2 = text_encoder_2_inference.create_flux_text_encoder_2(
        configuration.nunchaku_t5_model_path,
        configuration)

    assert text_encoder_2 is not None

    pipeline = text_encoder_2_inference.create_flux_text_encoder_2_pipeline(
        configuration.flux_model_path,
        configuration,
        text_encoder_2)

    change_pipe_to_cuda_or_not(configuration, pipeline)

    assert pipeline.text_encoder is not None
    assert pipeline.text_encoder_2 == text_encoder_2
    assert pipeline.transformer is None
    assert pipeline.vae is None
