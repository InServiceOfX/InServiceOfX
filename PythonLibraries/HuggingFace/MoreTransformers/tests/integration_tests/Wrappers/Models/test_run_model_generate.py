from corecode.Utilities import DataSubdirectories
from pathlib import Path

from moretransformers.Wrappers.Models import run_model_generate
from moretransformers.Wrappers.Models import create_AutoModelForCausalLM

from transformers import AutoTokenizer, TextIteratorStreamer

import torch

from moretransformers.Configurations import GenerationConfiguration

data_sub_dirs = DataSubdirectories()

test_data_directory = Path(__file__).resolve().parents[3] / "TestData"

def test_run_model_generate_generates():
    pretrained_model_path = data_sub_dirs.ModelsLLM / "meta-llama" / \
        "Llama-3.2-1B-Instruct"

    model, _ = create_AutoModelForCausalLM(
        model_subdirectory=pretrained_model_path,
        #torch_dtype=torch.bfloat16,
        torch_dtype=torch.float16,
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

    input_ids = tokenizer.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        return_tensors="pt").to(model.device)

    streamer = TextIteratorStreamer(
        tokenizer,
        timeout=60.0,
        skip_prompt=True)

    generation_configuration = GenerationConfiguration(
        test_data_directory / "generation_configuration.yml")

    # tl;dr this isn't an OrderedDict, but a Tensor object.
    # See transformers/utils/generic.py for definition of
    # class ModelOutput(OrderedDict)
    # As a derived class of ModelOutput, see generic.py for its interface.
    # In generation/utils.py, GenerateDecoderOnlyOutput,
    # GenerateEncoderDecoderOutput, etc. are defined.
    # But, it's a Tensor object instead.
    generate_output = run_model_generate(
        input_ids=input_ids,
        model=model,
        streamer=streamer,
        eos_token_id=[128001,128008,128009],
        generation_configuration=generation_configuration)

    output_buffer = ""
    for new_text in streamer:
        output_buffer += new_text

    assert isinstance(output_buffer, str)
    assert output_buffer != ""
    assert "CSS" in output_buffer
