from nunchaku import NunchakuT5EncoderModel

def create_flux_text_encoder_2(directory_path, configuration):
    assert "svdq-flux.1-t5" in str(directory_path)
    text_encoder_2 = NunchakuT5EncoderModel.from_pretrained(
        directory_path,
        torch_dtype=configuration.torch_dtype)
    return text_encoder_2

from diffusers import FluxPipeline

def create_flux_text_encoder_2_pipeline(
    pretrained_flux_model_path,
    configuration,
    text_encoder_2):
    return FluxPipeline.from_pretrained(
        pretrained_flux_model_path,
        text_encoder_2=text_encoder_2,
        transformer=None,
        vae=None,
        torch_dtype=configuration.torch_dtype)

import torch

def encode_prompt(
    pipeline,
    generation_configuration,
    prompt,
    prompt2=None):
    kwargs = {}
    kwargs["max_sequence_length"] = generation_configuration.max_sequence_length
    kwargs["prompt"] = prompt
    # "prompt_2" has to have a value, even if it's None; otherwise, this error
    # is obtained:
    # TypeError: FluxPipeline.encode_prompt() missing 1 required positional argument: 'prompt_2'
    kwargs["prompt_2"] = prompt2
    with torch.no_grad():
        return pipeline.encode_prompt(**kwargs)

# TODO: This doesn't work to free up VRAM memory and may not work at all because
# the variables are at function scope and there may still be references to it in
# the input arguments.
def delete_variables_on_device(pipeline, text_encoder_2):
    del pipeline.text_encoder
    del pipeline.text_encoder_2
    del pipeline.tokenizer
    del pipeline.tokenizer_2
    del text_encoder_2