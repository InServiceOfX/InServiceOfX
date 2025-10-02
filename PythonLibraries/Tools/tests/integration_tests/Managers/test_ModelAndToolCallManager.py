from commonapi.Messages import (
    ConversationSystemAndPermanent,
    UserMessage)
from corecode.Utilities import DataSubdirectories, is_model_there

data_subdirectories = DataSubdirectories()
relative_model_path = "Models/LLM/Qwen/Qwen3-0.6B"
is_model_downloaded, model_path = is_model_there(
    relative_model_path,
    data_subdirectories)
model_is_not_downloaded_message = f"Model {relative_model_path} not downloaded"

from moretransformers.Applications import ModelAndTokenizer
from moretransformers.Configurations import (
    CreateDefaultGenerationConfigurations,
    FromPretrainedModelConfiguration,
    FromPretrainedTokenizerConfiguration,)
from moretransformers.Tools import ToolCallProcessor

import pytest, torch

from TestSetup.ToolsForTest import ToolsForTest
from tools.Managers import ModelAndToolCallManager

@pytest.mark.skipif(
        not is_model_downloaded, reason=model_is_not_downloaded_message)
def test_process_messages_once_for_any_tool_calls_with_Qwen3_and_single_tool():
    tools = [
        ToolsForTest.get_current_temperature,
        ToolsForTest.get_current_wind_speed]

    csp = ConversationSystemAndPermanent()
    csp.add_system_message(
        "You are a bot that responds to weather queries. You should "
        "reply with the unit used in the queried location.")
    csp.append_message(UserMessage(
        content="Hey, what's the temperature in Paris right now?"))

    tool_call_processor = ToolCallProcessor(
        available_functions={
            "get_current_temperature": ToolsForTest.get_current_temperature,
            "get_current_wind_speed": ToolsForTest.get_current_wind_speed
        })

    from_pretrained_tokenizer_configuration = FromPretrainedTokenizerConfiguration(
        pretrained_model_name_or_path=model_path)
    from_pretrained_model_configuration = FromPretrainedModelConfiguration(
        pretrained_model_name_or_path=model_path,
        device_map="cuda:0",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2")

    generation_configuration = \
        CreateDefaultGenerationConfigurations.for_Qwen3_thinking()
    # Note: If you *don't* set do_sample = True, you'll get this warning:
    # The following generation flags are not valid and may be ignored: ['temperature', 'top_p', 'min_p', 'top_k']. Set `TRANSFORMERS_VERBOSITY=info` for more details.
    generation_configuration.do_sample = True

    mat = ModelAndTokenizer(
        model_path,
        from_pretrained_model_configuration=from_pretrained_model_configuration,
        from_pretrained_tokenizer_configuration= \
            from_pretrained_tokenizer_configuration,
        generation_configuration=generation_configuration)

    mat.load_model()
    mat.load_tokenizer()

    matcm = ModelAndToolCallManager(
        mat,
        tool_call_processor)

    assert len(csp.get_conversation_as_list_of_dicts()) == 2

    process_once_results = matcm._process_messages_once_for_any_tool_calls(
        csp.get_conversation_as_list_of_dicts())

    assert len(csp.get_conversation_as_list_of_dicts()) == 2
    assert len(process_once_results) == 3
    assert process_once_results[0]
    assert len(process_once_results[1]) == 3
    assert process_once_results[1][2]["role"] == "assistant"
    assert len(process_once_results[2]) == 1

