from moretransformers.Configurations import Configuration

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Qwen2ForCausalLM,
    Qwen2Model,
    Qwen2TokenizerFast)

from pathlib import Path

import torch

test_data_directory = Path(__file__).resolve().parents[2] / "TestData"

configuration_qwen2math = Configuration(
    test_data_directory / "configuration-qwen2math.yml")


def test_constructs_with_AutoModelForCausalLM():
    model = AutoModelForCausalLM.from_pretrained(
        configuration_qwen2math.model_path,
        torch_dtype="auto",
        device_map="auto",
        local_files_only=True)
    
    assert isinstance(model, Qwen2ForCausalLM)
    assert isinstance(model.model, Qwen2Model)
    assert model._tied_weights_keys == ["lm_head.weight"]

def test_constructs_with_AutoTokenizer():
    tokenizer = AutoTokenizer.from_pretrained(
        configuration_qwen2math.model_path,
        local_files_only=True)

    assert isinstance(tokenizer, Qwen2TokenizerFast)

prompt = "Find the value of $x$ that satisfies the equation $4x+5 = 6x+7$."

# CoT Chain of Thought
chain_of_thought_messages = [
    {
        "role": "system",
        "content": "Please reason step by step, and put your final answer within \\boxed{}."
    },
    {"role": "user", "content": prompt}]
 
def test_tokenizer_applies_chat_template_to_chain_of_thought():
    tokenizer = AutoTokenizer.from_pretrained(
        configuration_qwen2math.model_path,
        local_files_only=True)
   
    text = tokenizer.apply_chat_template(
        chain_of_thought_messages,
        tokenize=False,
        add_generation_prompt=True)
    
    assert isinstance(text, str)

    expected = "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>user\nFind the value of $x$ that satisfies the equation $4x+5 = 6x+7$.<|im_end|>\n<|im_start|>assistant\n"
    assert text == expected

# TIR Tool-Integrated Reasoning
tool_integrated_reasoning_messages = [
    {
        "role": "system",
        "content": "Please integrate natural language reasoning with programs to solve the problem above, and put your final answer within \\boxed{}."
    },
    {"role": "user", "content": prompt}
]

def test_tokenizer_applies_chat_template_to_tool_integrated_reasoning():
    tokenizer = AutoTokenizer.from_pretrained(
        configuration_qwen2math.model_path,
        local_files_only=True)
    
    text = tokenizer.apply_chat_template(
        tool_integrated_reasoning_messages,
        tokenize=False,
        add_generation_prompt=True)
    
    assert isinstance(text, str)

    expected = "<|im_start|>system\nPlease integrate natural language reasoning with programs to solve the problem above, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>user\nFind the value of $x$ that satisfies the equation $4x+5 = 6x+7$.<|im_end|>\n<|im_start|>assistant\n"
    assert text == expected

def test_tokenizer_calls_on_chain_of_thought_text():
    tokenizer = AutoTokenizer.from_pretrained(
        configuration_qwen2math.model_path,
        local_files_only=True)
    
    text = tokenizer.apply_chat_template(
        chain_of_thought_messages,
        tokenize=False,
        add_generation_prompt=True)
    
    model_inputs = tokenizer([text], return_tensors="pt")
    assert "input_ids" in model_inputs.keys()
    assert isinstance(model_inputs['input_ids'], torch.Tensor)

def test_tokenizer_calls_on_tool_integrated_reasoning_text():
    tokenizer = AutoTokenizer.from_pretrained(
        configuration_qwen2math.model_path,
        local_files_only=True)
    
    text = tokenizer.apply_chat_template(
        tool_integrated_reasoning_messages,
        tokenize=False,
        add_generation_prompt=True)
    
    model_inputs = tokenizer([text], return_tensors="pt")
    assert "input_ids" in model_inputs.keys()
    assert isinstance(model_inputs['input_ids'], torch.Tensor)

def test_model_calls_on_chain_of_thought_input_ids():
    model = AutoModelForCausalLM.from_pretrained(
        configuration_qwen2math.model_path,
        torch_dtype="auto",
        device_map="auto",
        local_files_only=True)

    tokenizer = AutoTokenizer.from_pretrained(
        configuration_qwen2math.model_path,
        local_files_only=True)
    
    text = tokenizer.apply_chat_template(
        chain_of_thought_messages,
        tokenize=False,
        add_generation_prompt=True)
    
    # The call to .to(model.device) is important because otherwise this error is
    # obtained:
    # RuntimeError: Expected all tensors to be on the same device, but found at
    # least two devices, cuda:0 and cpu! (when che..
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(**model_inputs, max_new_tokens=512)
    assert isinstance(generated_ids, torch.Tensor)

    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    assert isinstance(response, list)
    assert isinstance(response[0], str)
    # Un
    #print(response)

def test_model_calls_on_tool_integrated_reasoning_input_ids():
    model = AutoModelForCausalLM.from_pretrained(
        configuration_qwen2math.model_path,
        torch_dtype="auto",
        device_map="auto",
        local_files_only=True)

    tokenizer = AutoTokenizer.from_pretrained(
        configuration_qwen2math.model_path,
        local_files_only=True)
    
    text = tokenizer.apply_chat_template(
        tool_integrated_reasoning_messages,
        tokenize=False,
        add_generation_prompt=True)
    
    # The call to .to(model.device) is important because otherwise this error is
    # obtained:
    # RuntimeError: Expected all tensors to be on the same device, but found at
    # least two devices, cuda:0 and cpu! (when che..
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(**model_inputs, max_new_tokens=512)

    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(model_inputs.input_ids,generated_ids)]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    assert isinstance(response, list)
    assert isinstance(response[0], str)
    # Uncomment to see the response
    #print(response)