from corecode.Utilities import DataSubdirectories, is_model_there

from morediffusers.Applications import FluxNunchakuAndLoRAs
from morediffusers.Configurations import (
    FluxGenerationConfiguration,
    PipelineInputs,
    NunchakuConfiguration,
    NunchakuLoRAsConfiguration)

import pytest

data_subdirectories = DataSubdirectories()

relative_model_path = "Models/Diffusion/black-forest-labs/FLUX.1-dev"

is_model_downloaded, model_path = is_model_there(
    relative_model_path,
    data_subdirectories)

relative_nunchaku_mode_path = "Models/Diffusion/jib-mix-svdq"

is_nunchaku_model_downloaded, nunchaku_model_path = is_model_there(
    relative_nunchaku_mode_path,
    data_subdirectories)

relative_nunchaku_t5_model_path = "Models/Diffusion/mit-han-lab/svdq-flux.1-t5"

is_nunchaku_t5_model_downloaded, nunchaku_t5_model_path = is_model_there(
    relative_nunchaku_t5_model_path,
    data_subdirectories)


@pytest.mark.skipif(
    not is_model_downloaded or \
        not is_nunchaku_model_downloaded or \
        not is_nunchaku_t5_model_downloaded,
    reason="Models not downloaded")
def test_FluxNunchakuAndLoRAs_creates_prompt_embeds():
    configuration = NunchakuConfiguration(
        flux_model_path=model_path,
        nunchaku_model_path=nunchaku_model_path,
        nunchaku_t5_model_path=nunchaku_t5_model_path,
    )
    configuration.cuda_device = "cuda:0"
    configuration.torch_dtype = "bfloat16"

    generation_configuration = FluxGenerationConfiguration()
    pipeline_inputs = PipelineInputs()
    loras_configuration = NunchakuLoRAsConfiguration()

    flux_nunchaku_and_loras = FluxNunchakuAndLoRAs(
        configuration,
        generation_configuration,
        pipeline_inputs,
        loras_configuration)

    prompt_embeds, pooled_prompt_embeds, text_ids, negative_prompt_embeds, negative_pooled_prompt_embeds, negative_text_ids = \
        flux_nunchaku_and_loras.create_prompt_embeds()

    flux_nunchaku_and_loras.delete_text_encoder_2_and_pipeline()

    print(type(prompt_embeds))
    print(type(pooled_prompt_embeds))
    print(type(text_ids))
    print(type(negative_prompt_embeds))
    print(type(negative_pooled_prompt_embeds))
    print(type(negative_text_ids))

    assert prompt_embeds is not None
    assert pooled_prompt_embeds is not None
    assert text_ids is not None
    assert negative_prompt_embeds is not None
    assert negative_pooled_prompt_embeds is not None
    assert negative_text_ids is not None

    assert len(flux_nunchaku_and_loras._prompt_embeds) == 1
    assert len(flux_nunchaku_and_loras._pooled_prompt_embeds) == 1
    assert len(flux_nunchaku_and_loras._negative_prompt_embeds) == 1
    assert len(flux_nunchaku_and_loras._negative_pooled_prompt_embeds) == 1
    assert len(flux_nunchaku_and_loras._corresponding_prompts) == 1

@pytest.mark.skipif(
    not is_model_downloaded or \
        not is_nunchaku_model_downloaded or \
        not is_nunchaku_t5_model_downloaded,
    reason="Models not downloaded")
def test_FluxNunchakuAndLoRAs_call_pipeline_works():
    configuration = NunchakuConfiguration(
        flux_model_path=model_path,
        nunchaku_model_path=nunchaku_model_path,
        nunchaku_t5_model_path=nunchaku_t5_model_path,
    )
    configuration.cuda_device = "cuda:0"
    configuration.torch_dtype = "bfloat16"

    generation_configuration = FluxGenerationConfiguration()
    generation_configuration.height = 1216
    generation_configuration.width = 832
    generation_configuration.num_inference_steps = 30
    generation_configuration.seed = 2652086967
    generation_configuration.guidance_scale = 5.5
    generation_configuration.max_sequence_length = 512

    pipeline_inputs = PipelineInputs()
    pipeline_inputs.prompt = "a photo of an astronaut riding a horse"
    pipeline_inputs.negative_prompt = "cartoon, drawing, painting, illustration"

    pipeline_inputs.prompt_2 = "a photo of a cat"
    pipeline_inputs.negative_prompt_2 = \
        "cartoon, drawing, painting, illustration"

    loras_configuration = NunchakuLoRAsConfiguration()

    flux_nunchaku_and_loras = FluxNunchakuAndLoRAs(
        configuration,
        generation_configuration,
        pipeline_inputs,
        loras_configuration)

    prompt_embeds, pooled_prompt_embeds, text_ids, negative_prompt_embeds, negative_pooled_prompt_embeds, negative_text_ids = \
        flux_nunchaku_and_loras.create_prompt_embeds()

    flux_nunchaku_and_loras.delete_text_encoder_2_and_pipeline()

    flux_nunchaku_and_loras.create_transformer_and_pipeline()

    images = flux_nunchaku_and_loras.call_pipeline(
        prompt_embeds,
        pooled_prompt_embeds,
        negative_prompt_embeds,
        negative_pooled_prompt_embeds)

    assert images is not None