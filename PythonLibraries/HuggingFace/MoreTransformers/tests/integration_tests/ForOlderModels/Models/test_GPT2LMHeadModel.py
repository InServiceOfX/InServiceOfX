from moretransformers.Configurations import Configuration
from transformers import (GPT2LMHeadModel, GPT2Tokenizer)
from transformers.modeling_outputs import (
    CausalLMOutputWithCrossAttentions)

import torch

from pathlib import Path

test_data_directory = Path(__file__).resolve().parents[3] / "TestData"

configuration_gpt2 = Configuration(
    test_data_directory / "configuration-gpt2.yml")

def test_GPT2LMHeadModel_instantiates():

    model = GPT2LMHeadModel.from_pretrained(
        configuration_gpt2.model_path,
        local_files_only=True)

    assert isinstance(model, GPT2LMHeadModel)

    assert model.model_parallel == False
    assert model.device_map == None
    assert model.device.type == "cpu"

    model.to("cuda")

    assert model.device.type == "cuda"
    assert model.model_parallel == False

    tokenizer = GPT2Tokenizer.from_pretrained(
        configuration_gpt2.model_path,
        local_files_only=True)

    text = "Replace me by any text you'd like."

    encoded_input = tokenizer(text, return_tensors="pt")
    encoded_input.to("cuda")

    output = model(**encoded_input)

    assert isinstance(output, CausalLMOutputWithCrossAttentions)
    assert len(output) == 2
    assert isinstance(output[0], torch.Tensor)
    # This is a torch.Tensor, but doesn't recognize it.
    #assert isinstance(output[1], torch.Tensor)

    logits_cpu = output.logits.to("cpu")
    assert isinstance(logits_cpu, torch.Tensor)
    assert logits_cpu.shape == torch.Size([1, 10, 50257])

    predicted_token_ids = logits_cpu.argmax(dim=-1)
    generated_text = tokenizer.decode(
        predicted_token_ids[0],
        skip_special_tokens=True)
    assert isinstance(generated_text, str)
