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

from moretransformers.Applications import UserInput

from moretransformers.Configurations import Configuration

from moretransformers.Wrappers.Pipelines import create_pipeline

def terminal_only_infinite_loop_text():

    configuration = Configuration()

    pipe = create_pipeline(
        configuration.model_path,
        task=configuration.task,
        torch_dtype=configuration.torch_dtype)

    while True:
        try:
            # Prompt for user input
            user_input = UserInput(configuration)

            #outputs = pipe(user_input.create_messages(), max_new_tokens=configuration.max_new_tokens)
            outputs = pipe(user_input.user_prompt.value)
            print(outputs[0]["generated_text"][-1])
            print(outputs[0]["generated_text"])
        except KeyboardInterrupt:
            print("\nProcess interrupted. Exiting...")
            clear_torch_cache_and_collect_garbage()
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            clear_torch_cache_and_collect_garbage()

    clear_torch_cache_and_collect_garbage()

if __name__ == "__main__":

    terminal_only_infinite_loop_text()