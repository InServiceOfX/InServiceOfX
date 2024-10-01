from corecode.Utilities import DataSubdirectories

from moretransformers.Wrappers.Models import create_AutoModelForCausalLM
from transformers import AutoTokenizer, PreTrainedTokenizerFast

import torch

import numpy as np

data_sub_dirs = DataSubdirectories()

def test_AutoTokenizer_instantiates():
    """
    AutoTokenizer is found in tokenization_auto.py, and is a generic tokenizer
    class that'll instantiate as one of the tokenizer classes with
    AutoTokenizer.from_pretrained(..). For Llama-3.2-1B-Instruct, the
    tokenizer_config.json, "tokenizer_class": "PreTrainedTokenizerFast"
    """
    pretrained_model_path = data_sub_dirs.ModelsLLM / "meta-llama" / \
        "Llama-3.2-1B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path)

    assert isinstance(tokenizer, PreTrainedTokenizerFast)
    # Found in class PreTrainedTokenizerFast(PreTrainedTokenizerBase), in
    # tokenization_utils_fast.py.
    assert tokenizer.is_fast
    assert tokenizer.vocab_size == 128000
    assert len(tokenizer.vocab.keys()) == 128256
    assert tokenizer.vocab[".^"] == 44359
    assert tokenizer.vocab["(chart"] == 63580
    assert tokenizer.vocab["Strange"] == 92434
    assert tokenizer.vocab["person"] == 9164
    assert tokenizer.vocab["rito"] == 29141
    assert tokenizer.vocab["(correct"] == 89064
    assert tokenizer.vocab["greso"] == 73256
    assert tokenizer.vocab["romium"] == 78959

    # From tokenization_utils_base.py, for class PreTrainedTokenizerBase
    assert tokenizer.max_len_single_sentence == 131071
    assert tokenizer.max_len_sentences_pair == 131070

def test_apply_chat_template_works():
    """
    Consider class PreTrainedTokenizerBase, method def apply_chat_template(
        self,
        conversation: Union[List[Dict[str, str]], List[List[Dict[str, str]]]],
        ...)
    with
    Args:
        conversation: A list of dicts with "role" and "content" keys,
        representing the chat history so far.
    """
    pretrained_model_path = data_sub_dirs.ModelsLLM / "meta-llama" / \
        "Llama-3.2-1B-Instruct"

    model, is_peft_available_variable = create_AutoModelForCausalLM(
        model_subdirectory=pretrained_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path)

    system_prompt = "You are a helpful assistant"
    message = \
        "Show me a code snippet of a website's sticky header in CSS and JavaScript."

    conversation = [
        {"role": "system", "content": system_prompt},
    ]
    conversation.append({"role": "user", "content": message})

    # See def apply_chat_template in tokenization_utils_base.py.
    # input_ids is of type
    # Union[str, List[int], List[str], List[List[int]], BatchEncoding]
    # A list of token ids representing the tokenized chat so far, including
    # control tokens. This output is ready to be passed to the model, either
    # directly or via methods like 'generate()'.
    input_ids = tokenizer.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        return_tensors="pt").to(model.device)

    input_ids_cpu = input_ids.to("cpu").numpy()

    assert isinstance(input_ids, torch.Tensor)
    assert input_ids.shape == (1, 56)
    assert input_ids_cpu[0][0] == 128000
    assert input_ids_cpu[0][1] == 128006
    assert input_ids_cpu[0][55] == 271

def test_apply_chat_template_works_with_return_dict():
    """
    Consider class PreTrainedTokenizerBase, method def apply_chat_template(
        self,
        conversation: Union[List[Dict[str, str]], List[List[Dict[str, str]]]],
        ...)
    with
    Args:
        conversation: A list of dicts with "role" and "content" keys,
        representing the chat history so far.
    """
    pretrained_model_path = data_sub_dirs.ModelsLLM / "meta-llama" / \
        "Llama-3.2-1B-Instruct"

    model, is_peft_available_variable = create_AutoModelForCausalLM(
        model_subdirectory=pretrained_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    model_device = next(model.parameters()).device

    assert model_device == torch.device('cuda', index=0)

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path)

    system_prompt = "You are a helpful assistant"
    message = \
        "Show me a code snippet of a website's sticky header in CSS and JavaScript."

    conversation = [
        {"role": "system", "content": system_prompt},
    ]
    conversation.append({"role": "user", "content": message})

    # See def apply_chat_template in tokenization_utils_base.py.
    # input_ids is of type
    # Union[str, List[int], List[str], List[List[int]], BatchEncoding]
    # A list of token ids representing the tokenized chat so far, including
    # control tokens. This output is ready to be passed to the model, either
    # directly or via methods like 'generate()'.
    # return_dict ('bool', defaults to 'False)
    # Whether to return a dictionary with named outputs. Has no effect if
    # tokenize is 'False'.
    return_output = tokenizer.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True).to(model.device)

    # TODO: This fails despite transformers tokenization_utils_base.py saying in
    # code comments it "will return a dict of tokenizer outputs instead."
    assert return_output.keys() == {"input_ids", "attention_mask"}
    
    input_ids_cpu = return_output["input_ids"].to("cpu").numpy()
    attention_mask_cpu = return_output["attention_mask"].to("cpu").numpy()

    assert isinstance(input_ids_cpu, np.ndarray)
    assert input_ids_cpu.shape == (1, 56)
    assert input_ids_cpu[0][0] == 128000
    assert input_ids_cpu[0][1] == 128006
    assert input_ids_cpu[0][55] == 271

    assert isinstance(attention_mask_cpu, np.ndarray)
    assert attention_mask_cpu.shape == (1, 56)
    assert attention_mask_cpu[0][0] == 1
    assert attention_mask_cpu[0][1] == 1
    assert attention_mask_cpu[0][55] == 1