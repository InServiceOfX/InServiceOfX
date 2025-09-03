from diffusers import FluxPipeline

def create_flux_transformer_pipeline(
    pretrained_flux_model_path,
    configuration,
    transformer):
    """
    This function creates a Flux pipeline with a transformer.

    Recall the function signature of __init__(..) for FluxPipeline,

    def __init__(
        self,
        ...
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        text_encoder_2: T5EncoderModel,
        tokenizer_2: T5TokenizerFast,
        transformer: FluxTransformer2DModel,
        image_encoder: CLIPVisionModelWithProjection = None
    )

    Recall that def from_pretrained(..) is inherited from DiffusionPipeline
    and it finally instantiates the pipeline class as
    model = pipeline_class(**init_kwargs)
    ...
    return model
    """
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

    if hasattr(generation_configuration, "true_cfg_scale") and \
        generation_configuration.true_cfg_scale is not None:
        kwargs["true_cfg_scale"] = generation_configuration.true_cfg_scale

    return kwargs

def call_pipeline(
    pipeline,
    prompt_embeds,
    pooled_prompt_embeds,
    configuration,
    generation_configuration,
    negative_prompt_embeds=None,
    negative_pooled_prompt_embeds=None):
    """
    In pipeline_flux.py, class FluxPipeline(..), recall the function signature
    of def __call__(..),
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        ...
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        ...
        guidance_scale: float = 3.5,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        ...
        prompt_embeds: Optional[torch.FloatTensor] = None,
        ...
        ):
    """
    kwargs = _get_pipeline_call_kwargs(configuration, generation_configuration)
    if negative_prompt_embeds is not None:
        kwargs["negative_prompt_embeds"] = negative_prompt_embeds
    if negative_pooled_prompt_embeds is not None:
        kwargs["negative_pooled_prompt_embeds"] = negative_pooled_prompt_embeds

    return pipeline(
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        **kwargs)

def delete_variables_on_device(pipeline, transformer):
    """
    TODO: This function was found not to delete the variables since maybe it's
    because the input variables serve as references.
    """
    del pipeline.transformer
    del pipeline
    del transformer
