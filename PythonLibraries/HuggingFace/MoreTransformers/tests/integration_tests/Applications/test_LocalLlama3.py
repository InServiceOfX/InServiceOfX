from moretransformers.Applications import LocalLlama3
from moretransformers.Configurations import Configuration, GenerationConfiguration

from pathlib import Path

test_data_directory = Path(__file__).resolve().parents[2] / "TestData"

configuration_llama3 = Configuration(
    test_data_directory / "configuration-llama3.yml")

generation_configuration_llama3 = GenerationConfiguration(
    test_data_directory / "generation_configuration-llama3.yml")

def test_LocalLlama3_generates_multiple_times():
    import copy
    import torch

    configuration = copy.deepcopy(configuration_llama3)
    configuration.device_map = "cuda:0"
    configuration.torch_dtype = torch.bfloat16
    generation_configuration = copy.deepcopy(generation_configuration_llama3)
    generation_configuration.do_sample = True

    local_engine = LocalLlama3(
        configuration,
        generation_configuration)

    assert len(local_engine.conversation_history.messages) == 0
    assert len(local_engine.conversation_history.content_hashes) == 0

    system_message_content = (
        "You are a helpful and informative AI assistant specializing in "
        "technology. Always provide detailed and accurate responses in a "
        "professional tone")
    
    local_engine.set_system_message(system_message_content)

    assert len(local_engine.conversation_history.messages) == 1
    assert len(local_engine.conversation_history.content_hashes) == 1

    user_message_content = (
        "Generate a list of SEO keywords for an Italian restaurant in "
        "New York City.")
    
    response = local_engine.generate_from_single_user_content(
        user_message_content)

    print("response:", response)

    assert len(local_engine.conversation_history.messages) == 3
    assert len(local_engine.conversation_history.content_hashes) == 3

    user_message_content = (
        "Act as if you are my personal trainer. Create a recipe that helps "
        "refuel after a workout using chicken, tomatoes, and carbs but "
        "excluding wheat.")
    
    response = local_engine.generate_from_single_user_content(
        user_message_content)

    print("response:", response)

    assert len(local_engine.conversation_history.messages) == 5
    assert len(local_engine.conversation_history.content_hashes) == 5
    
    local_engine.clear_conversation_history()

    assert len(local_engine.conversation_history.messages) == 0
    assert len(local_engine.conversation_history.content_hashes) == 0
