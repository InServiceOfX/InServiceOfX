"""
Instructions for running these tests:

e.g.
pytest -s ./integration_tests/test_QuickStartGuide.py -k "_run_offline_inference_with_LLM_API"
"""
from corecode.Utilities import DataSubdirectories, is_model_there
data_subdirectories = DataSubdirectories()
relative_llm_model_path = "Models/LLM/Qwen/Qwen3-0.6B"
is_llm_model_downloaded, llm_model_path = is_model_there(
    relative_llm_model_path,
    data_subdirectories)
llm_model_is_not_downloaded_message = \
    f"Model {relative_llm_model_path} not downloaded"

from tensorrt_llm import LLM, SamplingParams
from pathlib import Path

def test_LLM_from_tensorrt_llm_inits():

    assert isinstance(llm_model_path, Path)
    llm = LLM(model=llm_model_path)
    assert True

def test_run_offline_inference_with_LLM_API():
    """
    Reference:
    https://nvidia.github.io/TensorRT-LLM/quick-start-guide.html
    """
    llm = LLM(model=llm_model_path)

    # Sample prompts.
    prompts = [
        "Hello, my name is",
        "The capital of France is",
        "The future of AI is",
    ]

    # Create a sampling params.
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

    results = []

    for output in llm.generate(prompts, sampling_params):
        results.append(
            f"Prompt: {output.prompt!r}, Generated text: {output.outputs[0].text!r}")

    assert len(results) == 3
    assert len(results[0]) > 0
    assert len(results[1]) > 0
    assert len(results[2]) > 0

    for result in results:
        print(result)

import asyncio, pytest

@pytest.mark.asyncio
async def test_generate_text_asynchronously():
    """
    Reference:
    https://nvidia.github.io/TensorRT-LLM/examples/llm_inference_async.html#generate-text-asynchronously
    """

    llm = LLM(model=llm_model_path)

    # Sample prompts.
    prompts = [
        "Hello, my name is",
        "The capital of France is",
        "The future of AI is",
    ]

    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

    results = []

    async def task(prompt: str):
        output = await llm.generate_async(prompt, sampling_params)
        result = \
            f"Prompt: {output.prompt!r}, Generated text: {output.outputs[0].text!r}"
        results.append(result)
        print(result)

    # In pytest with @pytest.mark.asyncio, the event loop is already running
    # So we just await directly - no need for asyncio.run()
    tasks = [task(prompt) for prompt in prompts]
    await asyncio.gather(*tasks)

    assert len(results) == 3
    assert len(results[0]) > 0
    assert len(results[1]) > 0
    assert len(results[2]) > 0

    for result in results:
        print(result)

@pytest.mark.asyncio
async def test_generate_text_in_streaming():
    """
    Reference:
    https://nvidia.github.io/TensorRT-LLM/examples/llm_inference_async_streaming.html#generate-text-in-streaming
    """

    llm = LLM(model=llm_model_path)

    # Sample prompts.
    prompts = [
        "Hello, my name is",
        "The capital of France is",
        "The future of AI is",
    ]

    # Create a sampling params.
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

    results = []

    # Async based on Python coroutines
    async def task(id: int, prompt: str):
        # streaming=True is used to enable streaming generation.
        async for output in llm.generate_async(
            prompt,
            sampling_params,
            streaming=True):
            output_text = \
                f"Generation for prompt-{id}: {output.outputs[0].text!r}"
            results.append(output_text)
            print(output_text)

    tasks = [task(id, prompt) for id, prompt in enumerate(prompts)]
    await asyncio.gather(*tasks)

    assert len(results) > 0
    print(len(results))
    for result in results:
        assert len(result) > 0

    for result in results:
        print(result)

from tensorrt_llm.llmpi import GuidedDecodingParams

import json

def test_generate_text_with_guided_decoding():
    # Specify the guided decoding backend: xgrammar and llguidance are supported
    # currently.
    llm = LLM(model=llm_model_path, guided_decoding_backend='xgrammar'))

    schema_dict = {
        "title": "WirelessAccessPoint",
        "type": "object",
        "properties": {
            "ssid": {
                "title": "SSID",
                "type": "string"
            },
            "securityProtocol": {
                "title": "SecurityProtocol",
                "type": "string"
            },
            "bandwidth": {
                "title": "Bandwidth",
                "type": "string"
            }
        },
        "required": ["ssid", "securityProtocol", "bandwidth"]
    }

    # Convert to JSON string
    schema = json.dumps(schema_dict)

    prompt = [{
        'role': 'system',
        'content': (
            "You are a helpful assistant that answers in JSON. Here's the json "
            "schema you must adhere to:\n<schema>\n{'title': 'WirelessAccessPoint', 'type': 'object', 'properties': {'ssid': {'title': 'SSID', 'type': 'string'}, 'securityProtocol': {'title': 'SecurityProtocol', 'type': 'string'}, 'bandwidth': {'title': 'Bandwidth', 'type': 'string'}}, 'required': ['ssid', 'securityProtocol', 'bandwidth']}\n</schema>\n"

    },
    {
        'role': 'user',

        'content':

        "I'm currently configuring a wireless access point for our office network and I need to generate a JSON object that accurately represents its settings. The access point's SSID should be 'OfficeNetSecure', it uses WPA2-Enterprise as its security protocol, and it's capable of a bandwidth of up to 1300 Mbps on the 5 GHz band. This JSON object will be used to document our network configurations and to automate the setup process for additional access points in the future. Please provide a JSON object that includes these details."

    }]    