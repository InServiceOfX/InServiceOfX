def run_model_generate(
    input_ids,
    model,
    eos_token_id,
    streamer=None,
    generation_configuration=None,
    generation_config=None,
    attention_mask=None):
    """
    In generation/utils.py, class GenerationMixin:
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        streamer: Optional["BaseStreamer"] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]

    inputs ('torch.Tensor' of varying shape depending on the modality,
    *optional*): The sequence used as a prompt for generation or as model
    inputs to the encoder. If 'None' the method initializes it with
    'bos_token_id' and batch size of 1. For decoder-only models 'inputs'
    should be in the format of 'inputs_ids'. For encoder-decoder models
    *inputs* can represent any of 'input_ids', 'input_values',
    'input_features', or 'pixel_values'.

    generation_config ('GenerationConfig', *optional*):
    The generation configuration to be used as base parametrization for the
    generation call. '**kwargs' passed to generate matching attributes of
    'generation_config' will override them. If 'generation_config' isn't
    provided, default will be used, which has following loading priority:
    1) from the 'generation_config.json' model file, if it exists; 2) from
    the model configuration. Please note that unspecified parameters will
    inherit ['~generation.GenerationConfig']'s default value, whose
    documentation should be checked to parameterize generation.
    """
    # Create base generation config from model if none provided
    if generation_config is None and generation_configuration is None:
        generation_config = model.generation_config

    if generation_configuration is not None:
        # Set pad_token_id to eos_token_id if not set
        pad_token_id = getattr(model.config, 'pad_token_id', None)
        if pad_token_id is None:
            pad_token_id = eos_token_id

        generation_kwargs = {
            'max_new_tokens': generation_configuration.max_new_tokens,
            'do_sample': False if generation_configuration.temperature == 0 else True,
            'top_k': generation_configuration.top_k,
            'top_p': generation_configuration.top_p,
            'temperature': generation_configuration.temperature,
            'eos_token_id': eos_token_id,
            'pad_token_id': pad_token_id,
            'streamer': streamer,
            'repetition_penalty': 1.1,  # Add repetition penalty
            'use_cache': True,  # Enable KV caching
        }

        if attention_mask is not None:
            generation_kwargs['attention_mask'] = attention_mask

        return model.generate(
            input_ids=input_ids,
            **generation_kwargs
        )
    
    # Fallback to basic generation config
    kwargs = {'generation_config': generation_config, 'streamer': streamer}
    if attention_mask is not None:
        kwargs['attention_mask'] = attention_mask
        
    return model.generate(input_ids=input_ids, **kwargs)