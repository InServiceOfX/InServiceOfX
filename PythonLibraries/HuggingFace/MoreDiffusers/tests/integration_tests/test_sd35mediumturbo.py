from corecode.Utilities import DataSubdirectories, is_model_downloaded

from pathlib import Path

from diffusers import StableDiffusion3Pipeline
from transformers import T5EncoderModel, BitsAndBytesConfig
import pytest
import torch

data_sub_dirs = DataSubdirectories()

relative_model_path = \
    "Models/Diffusion/tensorart/stable-diffusion-3.5-medium-turbo"

is_model_downloaded, model_path = is_model_there(
    relative_model_path,
    data_subdirectories)

model_not_downloaded_message = f"Model not downloaded: {relative_model_path}"

@pytest.mark.skipif(
        not is_model_downloaded,
        reason=model_not_downloaded_message)
def test_sd35mediumturbo_and_StableDiffusion3Pipeline_from_pretrained_works():
    pipe = StableDiffusion3Pipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
        local_files_only=True
    )
    pipe = pipe.to("cuda")

@pytest.mark.skipif(
        not is_model_downloaded,
        reason=model_not_downloaded_message)
def test_sd35mediumturbo_and_StableDiffusion3Pipeline_from_pretrained_no_transformer_no_gpu():
    pipe = StableDiffusion3Pipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        transformer=None,
        vae=None,
        scheduler=None,
        use_safetensors=True,
        local_files_only=True
    )
    assert pipe.transformer is None
    assert pipe.vae is None
    assert pipe.text_encoder_3 is not None
    assert pipe.tokenizer_3 is not None
    assert pipe.text_encoder_2 is not None
    assert pipe.tokenizer_2 is not None
    assert pipe.text_encoder is not None
    assert pipe.tokenizer is not None

@pytest.mark.skipif(
        not is_model_downloaded,
        reason=model_not_downloaded_message)
def test_sd35mediumturbo_BitsAndBytesConfig_8bit_on_text_encoder_3():
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    # See transformers, modeling_utils.py, class PreTrainedModel,
    # def from_pretrained(..),
    text_encoder_3 = T5EncoderModel.from_pretrained(
        model_path,
        # A dictionary of configuration parameters or a QuantizationConfigMixin
        # object for quantization (e.g. bitsandbytes, gptq). There may be other
        # quantization-related kwargs, including load_in_4bit, and load_in_8bit,
        # which are parsed by QauntizationConfigParser. Supported only for
        # bitsandbytes quantization and not preferred.
        quantization_config=quantization_config,
        # subfolder (str, optional, defaults to "")
        # In case relevant files are located inside a subfolder of the model
        # repo on huggingface.co.
        subfolder="text_encoder_3",
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
        local_files_only=True,
        # torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 80.00 MiB. GPU 0 has a total capacity of 7.69 GiB of which 24.94 MiB is free. Including non-PyTorch memory, this process has 0 bytes memory in use. Of the allocated memory 6.55 GiB is allocated by PyTorch, and 402.64 MiB is reserved by PyTorch but unallocated
        #device_map="cuda:0")
        )

prompt_1 = (
    "A beautiful bald girl with silver and white futuristic metal face "
    "jewelry, her full body made of intricately carved liquid glass in the "
    "style of Tadashi, the complexity master of cyberpunk, in the style of "
    "James Jean and Peter Mohrbacher. This concept design is trending on "
    "Artstation, with sharp focus, studio-quality photography, and highly "
    "detailed, intricate details.",)

negative_prompt_1 = "ugly, bad, low quality, low resolution, blurry"

@pytest.mark.skipif(
        not is_model_downloaded,
        reason=model_not_downloaded_message)
def test_sd35mediumturbo_BitsAndBytesConfig_4bit_on_text_encoder_3():
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16)

    # See transformers, modeling_utils.py, class PreTrainedModel,
    # def from_pretrained(..),
    text_encoder_3 = T5EncoderModel.from_pretrained(
        model_path,
        # A dictionary of configuration parameters or a QuantizationConfigMixin
        # object for quantization (e.g. bitsandbytes, gptq). There may be other
        # quantization-related kwargs, including load_in_4bit, and load_in_8bit,
        # which are parsed by QauntizationConfigParser. Supported only for
        # bitsandbytes quantization and not preferred.
        quantization_config=quantization_config,
        # subfolder (str, optional, defaults to "")
        # In case relevant files are located inside a subfolder of the model
        # repo on huggingface.co.
        subfolder="text_encoder_3",
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
        local_files_only=True,
        #device_map="cuda:0")
        )

    pipe = StableDiffusion3Pipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        transformer=None,
        vae=None,
        scheduler=None,
        use_safetensors=True,
        local_files_only=True,
        text_encoder_3=text_encoder_3,
    )

    assert pipe.transformer is None
    assert pipe.vae is None
    assert pipe.text_encoder_3 is not None
    assert pipe.tokenizer_3 is not None
    assert pipe.text_encoder_2 is not None
    assert pipe.tokenizer_2 is not None
    assert pipe.text_encoder is not None
    assert pipe.tokenizer is not None

    kwargs = {}
    kwargs["prompt"] = prompt_1
    kwargs["prompt_2"] = prompt_1
    kwargs["prompt_3"] = prompt_1

    pipe = pipe.to("cuda:0")
    prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(**kwargs)
    assert prompt_embeds.shape == (1, 77, 1024)

@pytest.mark.skipif(
        not is_model_downloaded,
        reason=model_not_downloaded_message)
def test_sd35mediumturbo_and_StableDiffusion3Pipeline_from_pretrained_no_transformer():
    pipe = StableDiffusion3Pipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        transformer=None,
        vae=None,
        use_safetensors=True,
        local_files_only=True,
        text_encoder_3=None,
        tokenizer_3=None,
    )
    pipe = pipe.to("cuda:0")
    assert pipe.transformer is None
    assert pipe.vae is None
    assert pipe.text_encoder_3 is None
    assert pipe.tokenizer_3 is None
    assert pipe.text_encoder_2 is not None
    assert pipe.tokenizer_2 is not None
    assert pipe.text_encoder is not None
    assert pipe.tokenizer is not None

@pytest.mark.skipif(
        not is_model_downloaded,
        reason=model_not_downloaded_message)
def test_sd35mediumturbo_encodes_prompts_1_and_2():
    pipe = StableDiffusion3Pipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        transformer=None,
        vae=None,
        use_safetensors=True,
        local_files_only=True,
        text_encoder_3=None,
        tokenizer_3=None,
    )
    pipe = pipe.to("cuda:0")



from optimum.quanto import freeze, qint4, quantize, quantization_map

from diffusers.models.transformers import SD3Transformer2DModel

def test_sd35mediumturbo_quantized_transformer():
    transformer = SD3Transformer2DModel.from_pretrained(
        pretrained_diffusion_model_name_or_path,
        subfolder="transformer",
        torch_dtype=torch.bfloat16
        )
    quantize(transformer, weights=qint4)
    freeze(transformer)

    transformer.save_pretrained("sd35mediumturbo-transformer-bfloat16-qint4")
    qmap = quantization_map(transformer)
    with open("quanto_qmap.json", "w", encoding="utf8") as f:
        json.dump(qmap, f, indent=4)
