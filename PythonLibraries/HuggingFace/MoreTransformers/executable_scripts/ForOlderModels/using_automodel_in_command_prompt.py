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

from transformers import AutoModelForCausalLM, AutoTokenizer

import torch

def using_automodel_in_command_prompt():

    configuration = Configuration(
        python_libraries_path.parent / "Configurations" / "HuggingFace" / \
            "MoreTransformers" / "configuration.yml")

    model = AutoModelForCausalLM.from_pretrained(
        configuration.model_path,
        local_files_only=True)

    model.to("cuda")

    tokenizer = AutoTokenizer.from_pretrained(
        configuration.model_path,
        local_files_only=True)

    chat_history = []

    while True:
        try:
            # Prompt for user input
            user_input = UserPromptInput()

            encoded_input = tokenizer.encode(
                user_input.user_prompt.value + tokenizer.eos_token,
                return_tensors="pt")

            # Move encoded_input to the same device as the model
            encoded_input = encoded_input.to("cuda")

            bot_input_ids = torch.cat(
                [torch.LongTensor(chat_history).to("cuda"), encoded_input],
                dim=-1)

            chat_history = model.generate(
                bot_input_ids,
                max_length=1000,
                pad_token_id=tokenizer.eos_token_id).tolist()

            response = tokenizer.decode(chat_history[0]).split("<|endoftext|>")
            response = [(response[i], response[i + 1]) for i in range(0, len(response) - 1, 2)]

            print("Next output: ", response)
            print("DialoGPT: {}".format(
                tokenizer.decode(
                    chat_history[0][bot_input_ids.shape[-1]:],
                    skip_special_tokens=True)))

        except KeyboardInterrupt:
            print("\nProcess interrupted. Exiting...")
            clear_torch_cache_and_collect_garbage()
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            clear_torch_cache_and_collect_garbage()

    clear_torch_cache_and_collect_garbage()

if __name__ == "__main__":

    using_automodel_in_command_prompt()
