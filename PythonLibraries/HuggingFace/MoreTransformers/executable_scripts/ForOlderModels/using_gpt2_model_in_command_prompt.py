from pathlib import Path
import sys

python_libraries_path = Path(__file__).resolve().parents[4]
corecode_directory = python_libraries_path / "CoreCode"
more_transformers_directory = \
    python_libraries_path / "HuggingFace" / "MoreTransformers"

if not str(corecode_directory) in sys.path:
    sys.path.append(str(corecode_directory))

if not str(more_transformers_directory) in sys.path:
    sys.path.append(str(more_transformers_directory))

from corecode.Utilities import clear_torch_cache_and_collect_garbage

from moretransformers.Applications.UserInput import UserPromptInput
from moretransformers.Configurations import Configuration

from transformers import GPT2LMHeadModel, GPT2Tokenizer

import torch

def using_gpt2_model_in_command_prompt():

    configuration = Configuration(
        python_libraries_path.parent / "Configurations" / "HuggingFace" / \
            "MoreTransformers" / "configuration.yml")

    model = GPT2LMHeadModel.from_pretrained(
        configuration.model_path,
        local_files_only=True)

    model.to("cuda")

    tokenizer = GPT2Tokenizer.from_pretrained(
        configuration.model_path,
        local_files_only=True)

    while True:
        try:
            # Prompt for user input
            user_input = UserPromptInput()

            encoded_input = tokenizer(
                user_input.user_prompt.value,
                return_tensors="pt")

            encoded_input.to("cuda")
            output = model(**encoded_input)

            logits_cpu = output.logits.to("cpu")
            predicted_token_ids = logits_cpu.argmax(dim=-1)
            generated_text = tokenizer.decode(
                predicted_token_ids[0],
                skip_special_tokens=True)
            print("Next output: ", generated_text)

        except KeyboardInterrupt:
            print("\nProcess interrupted. Exiting...")
            clear_torch_cache_and_collect_garbage()
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            clear_torch_cache_and_collect_garbage()

    clear_torch_cache_and_collect_garbage()

if __name__ == "__main__":

    using_gpt2_model_in_command_prompt()