from commonapi.Messages import (
    ConversationSystemAndPermanent,
    ParsePromptsCollection,
    AssistantMessage,
    UserMessage)

from corecode.FileIO import JSONFile
from corecode.Utilities import is_model_there, DataSubdirectories

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerFast,
    Qwen3ForCausalLM,
    Qwen2Tokenizer)

from pathlib import Path

import pytest
import torch

from moretransformers.Configurations import (
    CreateDefaultGenerationConfigurations,
    FromPretrainedModelConfiguration,
    FromPretrainedTokenizerConfiguration,
)

data_subdirectories = DataSubdirectories()

relative_model_path = "Models/LLM/Menlo/Lucy-128k"

is_model_downloaded, model_path = is_model_there(
    relative_model_path,
    data_subdirectories)

model_is_not_downloaded_message = f"Model {relative_model_path} not downloaded"

@pytest.mark.skipif(
        not is_model_downloaded, reason=model_is_not_downloaded_message)
def test_AutoModelForCausalLM_from_pretrained_works():
    model = AutoModelForCausalLM.from_pretrained(model_path)
    assert model is not None
    assert isinstance(model, Qwen3ForCausalLM)

def test_Qwen3ForCausalLM_from_pretrained_works():
    model = Qwen3ForCausalLM.from_pretrained(
        model_path,
        device_map="cuda:0",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2")
    assert model is not None
    assert isinstance(model, Qwen3ForCausalLM)

def test_AutoTokenizer_from_pretrained_works():
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    assert tokenizer is not None
    assert isinstance(tokenizer, PreTrainedTokenizerFast)

@pytest.mark.skipif(
        not is_model_downloaded, reason=model_is_not_downloaded_message)
def test_PreTrainedTokenizerFast_from_pretrained_works():
    # From tokenizer_utils_base.py class PreTrainedTokenizerbase
    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True)
    assert tokenizer is not None
    assert isinstance(tokenizer, PreTrainedTokenizerFast)

    assert tokenizer.chat_template is not None
    assert isinstance(tokenizer.chat_template, str)

    print(tokenizer.chat_template)

@pytest.mark.skipif(
        not is_model_downloaded, reason=model_is_not_downloaded_message)
def test_Qwen2Tokenizer_from_pretrained_works():
    # From tokenizer_utils_base.py class PreTrainedTokenizerbase
    tokenizer = Qwen2Tokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True)
    assert tokenizer is not None
    assert isinstance(tokenizer, Qwen2Tokenizer)

    assert tokenizer.chat_template is not None
    assert isinstance(tokenizer.chat_template, str)

@pytest.mark.skipif(
        not is_model_downloaded, reason=model_is_not_downloaded_message)
def test_generate_works():
    """
    I used examples from here:
    https://huggingface.co/LiquidAI/LFM2-1.2B

    There was no particular reason to use these examples.
    """
    tokenizer = Qwen2Tokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
        return_tensors='pt')

    model = Qwen3ForCausalLM.from_pretrained(
        model_path,
        device_map="cuda:0",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        # E       RuntimeError: FlashAttention only supports Ampere GPUs or newer.
        #attn_implementation="flash_attention_2")
        )

    prompt = "What is C. elegans?"
    input_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        return_tensors="pt",
        tokenize=True).to(model.device)

    output = model.generate(
        input_ids=input_ids,
        do_sample=True,
        # temperature, min_p, repetition_penalty suggested by
        # https://huggingface.co/LiquidAI/LFM2-1.2B
        temperature=0.3,
        min_p=0.15,
        repetition_penalty=1.05,
        max_new_tokens=512,
        )

    assert len(output) == 1

    print(
        "With special tokens: ",
        tokenizer.decode(output[0], skip_special_tokens=False))
    print(
        "Without special tokens: ",
        tokenizer.decode(output[0], skip_special_tokens=True))

@pytest.mark.skipif(
        not is_model_downloaded, reason=model_is_not_downloaded_message)
def test_generate_with_greater_new_tokens():
    """
    I used examples from here:
    https://huggingface.co/LiquidAI/LFM2-1.2B    

    There was no particular reason to use these examples.
    """
    tokenizer = Qwen2Tokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True)

    model = Qwen3ForCausalLM.from_pretrained(
        model_path,
        device_map="cuda:0",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        # E       RuntimeError: FlashAttention only supports Ampere GPUs or newer.
        #attn_implementation="flash_attention_2"
        )

    prompt = "What is C. elegans?"
    input_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        return_tensors="pt",
        tokenize=True).to(model.device)

    output = model.generate(
        input_ids,
        do_sample=True,
        # temperature, min_p, repetition_penalty suggested by
        # https://huggingface.co/LiquidAI/LFM2-1.2B
        temperature=0.3,
        min_p=0.15,
        repetition_penalty=1.05,
        max_new_tokens=65536
        )

    assert len(output) == 1

    print(
        "With special tokens: ",
        tokenizer.decode(output[0], skip_special_tokens=False))
    print(
        "Without special tokens: ",
        tokenizer.decode(output[0], skip_special_tokens=True))

@pytest.mark.skipif(
        not is_model_downloaded, reason=model_is_not_downloaded_message)
def test_generate_with_attention_mask():
    """
    I used examples from here:
    https://huggingface.co/LiquidAI/LFM2-1.2B    

    There was no particular reason to use these examples.
    """
    tokenizer = Qwen2Tokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True)

    model = Qwen3ForCausalLM.from_pretrained(
        model_path,
        device_map="cuda:0",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        # E       RuntimeError: FlashAttention only supports Ampere GPUs or newer.
        #attn_implementation="flash_attention_2"
        )

    prompt = "What is C. elegans?"
    prompt_str = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        # TODO: tokenize=False clearly returns a str so is return_tensors
        # needed?
        return_tensors="pt",
        tokenize=False)

    assert isinstance(prompt_str, str)

    encoded = tokenizer(prompt_str, return_tensors='pt', padding=True).to(
        model.device)

# E       AssertionError: assert False
# E        +  where False = isinstance({'input_ids': tensor([[151644,    872,    198,   3838,    374,    356,     13,  17720,    596,\n             30, 151645...   198]], device='cuda:0'), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], device='cuda:0')}, <class 'torch.Tensor'>)
# E        +    where <class 'torch.Tensor'> = torch.Tensor
    # TODO: Fix, error msg is above.
    #assert isinstance(encoded, Dict)

    encoded = {k: v.to(model.device) for k, v in encoded.items()}

    output = model.generate(
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        do_sample=True,
        # temperature, min_p, repetition_penalty suggested by
        # https://huggingface.co/LiquidAI/LFM2-1.2B
        temperature=0.9,
        min_p=0.15,
        repetition_penalty=1.05,
        max_new_tokens=65536
        )

    assert len(output) == 1

    print(
        "With special tokens: ",
        tokenizer.decode(output[0], skip_special_tokens=False))
    print(
        "Without special tokens: ",
        tokenizer.decode(output[0], skip_special_tokens=True))

@pytest.mark.skipif(
        not is_model_downloaded, reason=model_is_not_downloaded_message)
def test_use_configurations():
    """
    We've noticed above that we are repeating the generation configuration
    values. So we use a default generation configuration. Then we save off the
    conversation in JSON format.
    """
    from_pretrained_tokenizer_configuration = FromPretrainedTokenizerConfiguration(
        pretrained_model_name_or_path=model_path)
    tokenizer = Qwen2Tokenizer.from_pretrained(
        **from_pretrained_tokenizer_configuration.to_dict())

    from_pretrained_model_configuration = FromPretrainedModelConfiguration(
        pretrained_model_name_or_path=model_path,
        device_map="cuda:0",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2")
    model = Qwen3ForCausalLM.from_pretrained(
        **from_pretrained_model_configuration.to_dict())
    assert model.config.max_position_embeddings == 131072
    assert hasattr(model.config, "rope_scaling")

    generation_configuration = \
        CreateDefaultGenerationConfigurations.for_Menlo_Lucy_128k()

    user_message_texts = []

    if data_subdirectories.PromptsCollection.exists():
        parse_prompts_collection = ParsePromptsCollection(
            data_subdirectories.PromptsCollection)
        lines_of_files = parse_prompts_collection.load_manually_copied_X_posts()
        posts = parse_prompts_collection.parse_manually_copied_X_posts(
            lines_of_files)
        for post in posts:
            user_message_texts.append(post["prompt"])
    else:
        print(
            f"Prompts collection path not found in {data_subdirectories.PromptsCollection}")
        user_message_texts.append("What is C. elegans?")

    csap = ConversationSystemAndPermanent()

    for i in range(len(user_message_texts)):
        text = user_message_texts[i]
        print(text)
        csap.append_message(UserMessage(text))
        tokenizer_outputs = tokenizer.apply_chat_template(
            conversation=csap.get_conversation_as_list_of_dicts(),
            add_generation_prompt=True,
            return_tensors="pt",
            tokenize=True,
            return_dict=True).to(model.device)

        output = model.generate(
            input_ids=tokenizer_outputs["input_ids"],
            attention_mask=tokenizer_outputs["attention_mask"],
            **generation_configuration.to_dict())
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        assert isinstance(response, str)
        print(response)
        csap.append_message(AssistantMessage(response))

    JSONFile.save_json(
        Path.cwd() / "test_use_configurations.json",
        csap.get_conversation_as_list_of_dicts())
