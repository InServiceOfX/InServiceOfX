from nunchaku import NunchakuT5EncoderModel

def create_flux_text_encoder_2(directory_path, configuration):
    text_encoder_2 = NunchakuT5EncoderModel.from_pretrained(
        directory_path,
        torch_dtype=configuration.torch_dtype)
    return text_encoder_2

from diffusers import FluxPipeline, FluxControlPipeline

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

def create_flux_control_text_encoder_2_pipeline(
    pretrained_flux_model_path,
    configuration,
    text_encoder_2):
    return FluxControlPipeline.from_pretrained(
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
    prompt2=None,
    device=None,
    lora_scale=None):
    """
    Notice that in pipeline_flux.py, class FluxPipeline(..), def __call__(..),
    tracing through the use of the variable "do_true_cfg", it is set here,
    do_true_cfg = true_cfg_scale > 1 and has_neg_prompt
    and you see that self.encode_prompt(...) is called on negative prompts. We
    guess that encode_prompt(...) can also be used for negative prompts.
    """
    kwargs = {}
    kwargs["max_sequence_length"] = generation_configuration.max_sequence_length
    kwargs["prompt"] = prompt
    # "prompt_2" has to have a value, even if it's None; otherwise, this error
    # is obtained:
    # TypeError: FluxPipeline.encode_prompt() missing 1 required positional argument: 'prompt_2'
    kwargs["prompt_2"] = prompt2
    if device is not None:
        kwargs["device"] = torch.device(device)
    if lora_scale is not None:
        kwargs["lora_scale"] = float(lora_scale)

    with torch.no_grad():
        return pipeline.encode_prompt(**kwargs)

def flux_control_encode_prompt(
    pipeline,
    configuration,
    generation_configuration,
    prompt,
    prompt2="",
    lora_scale=None):
    kwargs = {}
    kwargs["max_sequence_length"] = generation_configuration.max_sequence_length
    kwargs["prompt"] = prompt
    # "prompt_2" has to have a value, even if it's ""; otherwise, this error
    # is obtained:
    # TypeError: FluxControlPipeline.encode_prompt() missing 1 required positional argument: 'prompt_2'
    kwargs["prompt_2"] = prompt2
    if configuration.cuda_device is not None:
        kwargs["device"] = torch.device(configuration.cuda_device)
    if generation_configuration.num_images_per_prompt is not None:
        kwargs["num_images_per_prompt"] = \
            generation_configuration.num_images_per_prompt
    if lora_scale is not None:
        kwargs["lora_scale"] = float(lora_scale)

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