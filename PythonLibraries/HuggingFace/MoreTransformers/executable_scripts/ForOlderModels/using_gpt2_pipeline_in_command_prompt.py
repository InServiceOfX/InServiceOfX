from pathlib import Path
import sys
import argparse  # Import argparse for command-line argument parsing

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
from transformers import pipeline, set_seed
import torch

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run GPT-2 model in command prompt.")
    parser.add_argument(
        '--num_return_sequences',
        type=int,
        default=5,
        help='Number of sequences to return (default: 5)'
    )
    return parser.parse_args()

def using_gpt2_pipeline_in_command_prompt(num_return_sequences):
    configuration = Configuration(
        python_libraries_path.parent / "Configurations" / "HuggingFace" / \
            "MoreTransformers" / "configuration.yml")
    
    pipeline_object = pipeline(
        task=configuration.task,
        model=configuration.model_path,
        device=torch.device("cuda")
    )

    set_seed(42)
    while True:
        try:
            user_input = UserPromptInput()

            output = pipeline_object(
                user_input.user_prompt.value,
                num_return_sequences=num_return_sequences
            )

            if isinstance(output, list):
                for element in output:
                    print("Response: ", element["generated_text"])
            else:
                print("Response: ", output)

        except KeyboardInterrupt:
            print("\nProcess interrupted. Exiting...")
            clear_torch_cache_and_collect_garbage()
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            clear_torch_cache_and_collect_garbage()

    clear_torch_cache_and_collect_garbage()

if __name__ == "__main__":
    args = parse_arguments()  # Parse command-line arguments
    using_gpt2_pipeline_in_command_prompt(args.num_return_sequences)
