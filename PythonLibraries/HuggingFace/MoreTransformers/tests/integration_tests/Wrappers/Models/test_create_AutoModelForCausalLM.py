from corecode.FileIO import is_directory_empty_or_missing
from corecode.Utilities import DataSubdirectories

from moretransformers.Wrappers.Models import create_AutoModelForCausalLM

from pathlib import Path
import pytest

# TODO: Consider if BitsAndBytesConfig should be tested here.
#from transformers import BitsAndBytesConfig, LlamaForCausalLM, LlamaModel

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaForCausalLM,
    LlamaModel)

import torch

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
    assert isinstance(tokenizer, PreTrainedTokenizerFast)
    model = AutoModelForCausalLM.from_pretrained(SMOL_V2_MODEL_DIR).to(device)
    assert isinstance(model, LlamaForCausalLM)
    assert isinstance(model.model, LlamaModel)
    assert model.device == torch.device(device, index=0)

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

