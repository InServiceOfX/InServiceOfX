from pathlib import Path
import sys

python_libraries_path = Path(__file__).resolve().parents[3]
corecode_directory = python_libraries_path / "CoreCode"
more_transformers_directory = \
    python_libraries_path / "HuggingFace" / "MoreTransformers"

if not str(corecode_directory) in sys.path:
    sys.path.append(str(corecode_directory))

if not str(more_transformers_directory) in sys.path:
    sys.path.append(str(more_transformers_directory))

from corecode.Utilities import clear_torch_cache_and_collect_garbage

from moretransformers.Applications.UserInput import (
    SystemPromptInput,
    UserPromptInput)
from moretransformers.Configurations import (
    Configuration,
    GenerationConfiguration)
from moretransformers.Conversation.StatementTypes import (
    SystemStatement,
    UserStatement,
    AssistantStatement)
from moretransformers.Wrappers.Models import (
    create_AutoModelForCausalLM,
    run_model_generate,
    set_model_to_cuda)

from transformers import AutoTokenizer, TextIteratorStreamer

import torch

def terminal_only_infinite_loop_instruct():

    configuration = Configuration(
        python_libraries_path.parent / "Configurations" / "HuggingFace" / \
            "MoreTransformers" / "configuration.yml")        

    model, _ = create_AutoModelForCausalLM(
        model_subdirectory=configuration.model_path,
        torch_dtype=configuration.torch_dtype,
        device_map="auto"
    )

    set_model_to_cuda(model)

    tokenizer = AutoTokenizer.from_pretrained(configuration.model_path)

    generation_configuration = GenerationConfiguration(
        python_libraries_path.parent / "Configurations" / "HuggingFace" / \
            "MoreTransformers" / "generation_configuration.yml")

    streamer = TextIteratorStreamer(
        tokenizer,
        timeout=generation_configuration.timeout,
        skip_prompt=True)

    system_prompt = SystemPromptInput()

    conversation = [
        SystemStatement(system_prompt.system_prompt.value).to_dict(),
    ]

    while True:
        try:
            # Prompt for user input
            user_input = UserPromptInput()

            conversation.append(
                UserStatement(user_input.user_prompt.value).to_dict())

            # See tokenization_utils_base.py in HuggingFace transformers.
            # add_generation_prompt=True
            # If this is set, prompt with token(s) that indicate start of an
            # assistant message will be appended to formatted output. This is
            # useful when you want to generate a response from the model. Note
            # that this argument will be passed to chat template, so it must be
            # supported in the template for this argument to have any effect.            
            return_output = tokenizer.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                return_tensors="pt",
                # See tokenization_utils_base.py in HuggingFace transformers for
                # implementation and code comments, which claims to return a
                # dict of tokenizer outputs.
                return_dict=True).to(model.device)

            # TODO: This didn't work to fix this issue:
            # inputs = tokenizer(input_ids, return_tensors="pt", padding=True)
            # attention_mask = inputs["attention_mask"]

            with torch.no_grad():
                generate_output = run_model_generate(
                    input_ids=return_output["input_ids"],
                    model=model,
                    streamer=streamer,
                    eos_token_id=generation_configuration.eos_token_id,
                    generation_configuration=generation_configuration,
                    attention_mask=return_output["attention_mask"])

            output_buffer = ""
            for new_text in streamer:
                output_buffer += new_text

            print("Next output: ", output_buffer)

            conversation.append(
                AssistantStatement(output_buffer).to_dict())

        except KeyboardInterrupt:
            print("\nProcess interrupted. Exiting...")
            clear_torch_cache_and_collect_garbage()
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            clear_torch_cache_and_collect_garbage()

    clear_torch_cache_and_collect_garbage()

if __name__ == "__main__":

    terminal_only_infinite_loop_instruct()