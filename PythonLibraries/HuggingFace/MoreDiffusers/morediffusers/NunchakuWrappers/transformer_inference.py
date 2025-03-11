from nunchaku import NunchakuFluxTransformer2dModel

def create_flux_transformer(directory_path):
    assert "svdq-int4-flux.1-dev" in str(directory_path)

    return NunchakuFluxTransformer2dModel.from_pretrained(directory_path)

from diffusers import FluxPipeline

def create_flux_transformer_pipeline(
    pretrained_flux_model_path,
    configuration,
    transformer):
    return FluxPipeline.from_pretrained(
        pretrained_flux_model_path,
        text_encoder=None,
        text_encoder_2=None,
        tokenizer=None,
        tokenizer_2=None,
        transformer=transformer,
        torch_dtype=configuration.torch_dtype)

from morediffusers.Wrappers import create_seed_generator

def _get_pipeline_call_kwargs(configuration, generation_configuration):
    kwargs = {}
    kwargs["num_inference_steps"] = generation_configuration.num_inference_steps
    kwargs["guidance_scale"] = generation_configuration.guidance_scale
    kwargs["height"] = generation_configuration.height
    kwargs["width"] = generation_configuration.width

    kwargs["generator"] = create_seed_generator(
        configuration,
        generation_configuration)

    return kwargs

def call_pipeline(
    pipeline,
    prompt_embeds,
    pooled_prompt_embeds,
    configuration,
    generation_configuration):
    return pipeline(
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        **_get_pipeline_call_kwargs(configuration, generation_configuration))

def delete_variables_on_device(pipeline, transformer):
    del pipeline.transformer
    del pipeline
    del transformer
