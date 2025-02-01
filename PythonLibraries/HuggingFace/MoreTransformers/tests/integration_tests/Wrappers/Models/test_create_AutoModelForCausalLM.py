from corecode.FileIO import is_directory_empty_or_missing
from corecode.Utilities import DataSubdirectories

from moretransformers.Wrappers.Models import create_AutoModelForCausalLM

import pytest

# TODO: Consider if BitsAndBytesConfig should be tested here.
#from transformers import BitsAndBytesConfig, LlamaForCausalLM, LlamaModel

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GPT2TokenizerFast,
    LlamaForCausalLM,
    LlamaModel)

import torch
import transformers

data_sub_dirs = DataSubdirectories()

SMOL_V2_MODEL_DIR = data_sub_dirs.ModelsLLM / "HuggingFaceTB" / \
    "SmolLM2-360M-Instruct"
SMOL_V2_skip_reason = f"Directory {SMOL_V2_MODEL_DIR} is empty or doesn't exist"

@pytest.mark.skipif(
    is_directory_empty_or_missing(SMOL_V2_MODEL_DIR),
    reason=SMOL_V2_skip_reason
)
def test_use_AutoModelForCausalLLM_with_SmolLMv2():
    device = "cuda"
    tokenizer = AutoTokenizer.from_pretrained(SMOL_V2_MODEL_DIR)
    assert isinstance(tokenizer, GPT2TokenizerFast)
    model = AutoModelForCausalLM.from_pretrained(SMOL_V2_MODEL_DIR).to(device)
    assert isinstance(model, LlamaForCausalLM)
    assert isinstance(model.model, LlamaModel)
    assert model.device == torch.device(device, index=0)

    messages = [{"role": "user", "content": "What is the capital of France."}]
    
    # Get both input_ids and attention_mask
    model_inputs = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        return_dict=True,
        add_generation_prompt=True).to(device)

    assert isinstance(model_inputs, transformers.tokenization_utils_base.BatchEncoding)

    outputs = model.generate(
        # This unpacks both input_ids and attention_mask
        **model_inputs,
        max_new_tokens=50,
        temperature=0.2,
        top_p=0.9,
        do_sample=True)

    assert isinstance(outputs, torch.Tensor)
    assert outputs.shape == torch.Size([1, 45])
    print(tokenizer.decode(outputs[0]))


def test_create_AutoModelForCausalLM_instantiates_without_quantization():
    pretrained_model_path = data_sub_dirs.ModelsLLM / "meta-llama" / \
        "Llama-3.2-1B-Instruct"

    model, is_peft_available_variable = create_AutoModelForCausalLM(
        model_subdirectory=pretrained_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    assert isinstance(model, LlamaForCausalLM)
    assert isinstance(model.model, LlamaModel)
    # TODO: Fix this
    #assert model.model.embed_tokens == torch.nn.Embedding(128256, 2048)
    assert isinstance(is_peft_available_variable, bool)
    # If the Docker build was correct, this should be True.
    assert is_peft_available_variable == True

    assert model.device == torch.device("cuda", index=0)

def test_create_AutoModelForCausalLM_instantiates_with_quantization():
    pretrained_model_path = data_sub_dirs.ModelsLLM / "meta-llama" / \
        "Llama-3.2-1B-Instruct"

    # TODO: Consider if BitsAndBytesConfig should be tested here.
    #quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    # model, is_peft_available_variable = create_AutoModelForCausalLM(
    #     model_subdirectory=pretrained_model_path,
    #     torch_dtype=torch.bfloat16,
    #     device_map="auto",
    #     quantization_config=quantization_config)

