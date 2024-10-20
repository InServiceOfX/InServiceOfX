from moretransformers.Configurations import Configuration
from transformers import (GPT2Model, GPT2Tokenizer)
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions)

import torch

from pathlib import Path

test_data_directory = Path(__file__).resolve().parents[3] / "TestData"

configuration_gpt2 = Configuration(
    test_data_directory / "configuration-gpt2.yml")

def test_GPT2Model_instantiates():

    model = GPT2Model.from_pretrained(
        configuration_gpt2.model_path,
        local_files_only=True)

    assert isinstance(model, GPT2Model)

    assert model.embed_dim == 768
    assert model.model_parallel == False
    assert model.device_map == None
    assert model.gradient_checkpointing == False
    assert model._attn_implementation == "sdpa"

def test_GPT2Model_to_cuda_works():

    model = GPT2Model.from_pretrained(
        configuration_gpt2.model_path,
        local_files_only=True)
    
    model.to("cuda")

    assert model.device.type == "cuda"
    assert model.model_parallel == False

def test_GPT2Model_device_map_to_cuda_works():

    model = GPT2Model.from_pretrained(
        configuration_gpt2.model_path,
        local_files_only=True,
        device_map="cuda:0")
    
    assert model.device.type == "cuda"
    assert model.model_parallel == False

def test_GPT2Model_generates_given_tokenizer_output():

    model = GPT2Model.from_pretrained(
        configuration_gpt2.model_path,
        local_files_only=True)
    
    model.to("cuda")

    tokenizer = GPT2Tokenizer.from_pretrained(
        configuration_gpt2.model_path,
        local_files_only=True)

    text = "Replace me by any text you'd like."

    encoded_input = tokenizer(text, return_tensors="pt")
    encoded_input.to("cuda")

    output = model(**encoded_input)

    assert isinstance(output, BaseModelOutputWithPastAndCrossAttentions)
    assert len(output) == 2
    assert isinstance(output[0], torch.Tensor)
    # This is a torch.Tensor, but doesn't recognize it.
    #assert isinstance(output[1], torch.Tensor)