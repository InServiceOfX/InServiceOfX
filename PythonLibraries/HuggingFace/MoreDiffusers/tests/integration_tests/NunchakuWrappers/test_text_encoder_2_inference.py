from morediffusers.Configurations import (
    NunchakuConfiguration)

from morediffusers.NunchakuWrappers import (
    text_encoder_2_inference,
    )

from morediffusers.Wrappers.pipelines import (
    change_pipe_to_cuda_or_not,
)

import torch

def test_create_flux_text_encoder_2_pipeline():
    configuration = NunchakuConfiguration(
        flux_model_path="/Data1/Models/Diffusion/black-forest-labs/FLUX.1-dev",
        nunchaku_model_path="/Data/Models/Diffusion/jib-mix-svdq",
    )
    configuration.nunchaku_t5_model_path = \
        "/Data1/Models/Diffusion/mit-han-lab/svdq-flux.1-t5"
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
