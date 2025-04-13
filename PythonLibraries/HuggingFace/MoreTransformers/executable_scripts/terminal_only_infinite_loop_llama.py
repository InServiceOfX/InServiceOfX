from pathlib import Path
import sys

python_libraries_path = Path(__file__).resolve().parents[3]
corecode_directory = python_libraries_path / "CoreCode"
more_transformers_directory = \
    python_libraries_path / "HuggingFace" / "MoreTransformers"
commonapi_directory = python_libraries_path / \
    "ThirdParties" / \
    "APIs" / \
    "CommonAPI"

if not str(corecode_directory) in sys.path:
    sys.path.append(str(corecode_directory))

if not str(more_transformers_directory) in sys.path:
    sys.path.append(str(more_transformers_directory))

if not str(commonapi_directory) in sys.path:
    sys.path.append(str(commonapi_directory))

from corecode.Utilities import clear_torch_cache_and_collect_garbage

from moretransformers.Applications import LocalLlama3
from moretransformers.Applications.UserInput import (
    SystemPromptInput,
    UserPromptInput)

from moretransformers.Configurations import (
    Configuration,
    GenerationConfiguration)

def terminal_only_infinite_loop_llama():

    configuration = Configuration(
        python_libraries_path.parent / "Configurations" / "HuggingFace" / \
            "MoreTransformers" / "configuration.yml")

    generation_configuration = GenerationConfiguration(
        python_libraries_path.parent / "Configurations" / "HuggingFace" / \
            "MoreTransformers" / "generation_configuration.yml")

    print("max_new_tokens:", generation_configuration.max_new_tokens)
    print("do_sample:", generation_configuration.do_sample)

    local_llama = LocalLlama3(
        configuration,
        generation_configuration)

    print("max_position_embeddings:",
          local_llama.llm_engine.model.config.max_position_embeddings)

    local_llama.clear_conversation_history()

    system_prompt_input = SystemPromptInput()

    local_llama.set_system_message(
        system_prompt_input.system_prompt.value)

    while True:
        try:
            user_prompt_input = UserPromptInput()

            response = local_llama.generate_from_single_user_content(
                user_prompt_input.user_prompt.value)

            print("response:", response)
            print("last conversation_history:",
                local_llama.conversation_history.messages[-1].role,
                local_llama.conversation_history.messages[-1].content)

        except KeyboardInterrupt:
            print("\nProcess interrupted. Exiting...")
            clear_torch_cache_and_collect_garbage()
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            clear_torch_cache_and_collect_garbage()

    clear_torch_cache_and_collect_garbage()

if __name__ == "__main__":
    terminal_only_infinite_loop_llama()
