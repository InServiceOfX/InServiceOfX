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
    ) -> Union[GenerateOuput, torch.LongTensor]

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
    if generation_configuration is not None:

        # Parameters that control the generation strategy used.
        # Whether or not to use sampling; use greedy decoding otherwise.
        do_sample = False if generation_configuration.temperature == 0 else True

        if attention_mask is not None:
            return model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=generation_configuration.max_new_tokens,
                do_sample=do_sample,
                top_k=generation_configuration.top_k,
                top_p=generation_configuration.top_p,
                temperature=generation_configuration.temperature,
                eos_token_id=eos_token_id,
                streamer=streamer)
        else:
            return model.generate(
                input_ids=input_ids,
                max_new_tokens=generation_configuration.max_new_tokens,
                do_sample=do_sample,
                top_k=generation_configuration.top_k,
                top_p=generation_configuration.top_p,
                temperature=generation_configuration.temperature,
                eos_token_id=eos_token_id,
                streamer=streamer)

    if attention_mask is not None:
        return model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
            streamer=streamer)
    else:
        return model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            streamer=streamer)