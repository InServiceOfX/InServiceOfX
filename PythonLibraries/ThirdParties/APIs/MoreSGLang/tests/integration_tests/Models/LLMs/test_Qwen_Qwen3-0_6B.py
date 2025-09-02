from corecode.Utilities import DataSubdirectories, is_model_there

from moresglang.Configurations import SamplingParameters
from transformers import AutoTokenizer
from typing import List, Dict

import pytest
import sglang

data_subdirectories = DataSubdirectories()

relative_model_path = "Models/LLM/Qwen/Qwen3-0.6B"

is_model_downloaded, model_path = is_model_there(
    relative_model_path,
    data_subdirectories)

model_is_not_downloaded_message = f"Model {relative_model_path} not downloaded"

@pytest.mark.skipif(
        not is_model_downloaded, reason=model_is_not_downloaded_message)
def test_offline_Engine_inits():
    llm = sglang.Engine(model_path=str(model_path), mem_fraction_static=0.8)
    assert True
    #print(dir(llm))
    #server_info = llm.get_server_info()
    #print(server_info)

@pytest.mark.skipif(
        not is_model_downloaded, reason=model_is_not_downloaded_message)
def test_offline_batch_inference():
    # https://docs.sglang.ai/basic_usage/offline_engine_api.html#Non-streaming-Synchronous-Generation
    llm = sglang.Engine(model_path=str(model_path), mem_fraction_static=0.75)

    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    sampling_params = {
        "temperature": 0.2,
        "top_p": 0.9,
    }

    responses = llm.generate(prompts, sampling_params)
    for prompt, output in zip(prompts, responses):
        print("===============================")
        print(f"Prompt: {prompt}\nGenerated text: {output['text']}")
    assert True

@pytest.mark.skipif(
        not is_model_downloaded, reason=model_is_not_downloaded_message)
def test_Engine_with_tokenizer():
    llm = sglang.Engine(
        model_path=str(model_path),
        # GPU exclusive.
        cpu_offload_gb=0,
        mem_fraction_static=0.75)

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    conversation = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "What's the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."},
        {"role": "user", "content": "Tell me more about Paris."}
    ]

    formatted_prompt = tokenizer.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=False)

    assert isinstance(formatted_prompt, str)

    sampling_params = SamplingParameters(
        max_new_tokens=32768,
        temperature=0.2,
        top_p=0.9,
    )

    sampling_param_dict = sampling_params.to_dict()

    response = llm.generate(
        formatted_prompt,
        sampling_params=sampling_param_dict)

    print(response.keys())
    #print(type(response))
    assert isinstance(response, dict)

    print(response["text"])

    # prompt = "What is C. elegans?"
    # input_ids = tokenizer.apply_chat_template(
    #     [{"role": "user", "content": prompt}],
    #     add_generation_prompt=True,
    #     tokenize=True)

from sglang import function, system, user, assistant, gen
from sglang.lang.api import set_default_backend

@function
def llm_sglang_frontend(s, history: List[Dict[str, str]]) -> str:
    for m in history:
        if m["role"] == "user":
            s += user(m["content"])
        elif m["role"] == "assistant":
            s += assistant(m["content"])
        elif m["role"] == "system":
            s += system(m["content"])
        else:
            raise ValueError(f"Unknown role: {m['role']}")
    s += assistant(gen("reply", max_tokens=32768))

# TODO: This test doesn't work: E       AttributeError: 'Engine' object has no attribute 'get_chat_template'
# @pytest.mark.skipif(
#         not is_model_downloaded, reason=model_is_not_downloaded_message)
# def test_Engine_with_sglang_frontend():
#     llm = sglang.Engine(
#         model_path=str(model_path),
#         mem_fraction_static=0.75)

#     set_default_backend(llm)

#     state = llm_sglang_frontend.run(
#         history=[
#             {"role":"system","content":"You are concise."},
#             {"role":"user","content":"Summarize CUDA graphs in one sentence."},
#         ],
#         sampling_params=SamplingParameters(
#             max_new_tokens=32768,
#             temperature=0.2,
#             top_p=0.9,
#         ).to_dict(),
#     )

#     print(state["reply"])

#     assert True

# TODO: ChatTemplate doesn't work with model_path argument.
#from sglang.lang.chat_template import ChatTemplate

# TODO: This test doesn't work: E       TypeError: Engine.generate() got an unexpected keyword argument 'format'
# @pytest.mark.skipif(
#         not is_model_downloaded, reason=model_is_not_downloaded_message)
# def test_Engine_with_format_chat():
#     conversation = [
#         {"role": "system", "content": "You are a helpful AI assistant."},
#         {"role": "user", "content": "What's the capital of France?"},
#         {"role": "assistant", "content": "The capital of France is Paris."},
#         {"role": "user", "content": "Tell me more about Paris."}
#     ]

#     llm = sglang.Engine(
#         model_path=str(model_path),
#         mem_fraction_static=0.75)

#     sampling_params = SamplingParameters(
#         max_new_tokens=32768,
#         temperature=0.2,
#         top_p=0.9,
#     )

#     sampling_param_dict = sampling_params.to_dict()

#     response = llm.generate(
#         conversation,
#         sampling_params=sampling_param_dict,
#         format="chat")

#     print(response["text"])

#     assert True