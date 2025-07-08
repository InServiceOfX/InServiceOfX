from commonapi.Messages import (
    AssistantMessage,
    ConversationAndSystemMessages,
    UserMessage)
from corecode.FileIO import JSONFile
from corecode.SetupProjectData.SetupPrompts import \
    ParseJujumilk3LeakedSystemPrompts
from corecode.Utilities import (
    get_environment_variable,
    load_environment_file,
    setup_datasets_path)
from moregroq.Wrappers import GroqAPIWrapper
from moregroq.Wrappers.ChatCompletionConfiguration \
    import ChatCompletionConfiguration
from moretransformers.Applications.Datasets import ParseOpenAssistantOasst1
from moretransformers.Wrappers.Datasets import LoadAndSaveLocally
from pathlib import Path

import pytest

load_environment_file()

def path_for_system_prompts_0():
    return ParseJujumilk3LeakedSystemPrompts()._repo_path

def path_for_example_dataset_0():
    datasets_path = setup_datasets_path()
    return datasets_path / "OpenAssistant" / "oasst1"

@pytest.fixture
def train_prompts_and_system_prompt():
    """Fixture that provides train_prompts and system_prompt for tests."""
    datasets_path = setup_datasets_path()
    load_and_save_locally = LoadAndSaveLocally(datasets_path)
    dataset_name = "OpenAssistant/oasst1"
    dataset = load_and_save_locally.load_from_disk(dataset_name)
    train_prompts = ParseOpenAssistantOasst1.parse_for_train_prompter(dataset)
    train_english_prompts = \
        ParseOpenAssistantOasst1.parse_for_train_prompter_english(dataset)

    parse_jujumilk3_leaked_system_prompts = ParseJujumilk3LeakedSystemPrompts()
    system_prompt, _, _ = \
        parse_jujumilk3_leaked_system_prompts.get_anthropic_claude_3_7_sonnet_prompt()
    
    return train_prompts, train_english_prompts, system_prompt

@pytest.mark.skipif(
    not path_for_system_prompts_0().exists() or \
        not path_for_example_dataset_0().exists(),
    reason=(
        "Repository jujumilk3/leaked-system-prompts or OpenAssistant/oasst1 "
        "not found locally"))
def test_GroqAPIWrapper_and_ConversationAndSystemMessages_works(
    train_prompts_and_system_prompt):
    _, train_english_prompts, system_prompt = \
        train_prompts_and_system_prompt

    chat_completion_configuration = ChatCompletionConfiguration()
    chat_completion_configuration.model = "deepseek-r1-distill-llama-70b"

    groq_api_wrapper = GroqAPIWrapper(
        api_key=get_environment_variable("GROQ_API_KEY"))

    conversation_and_system_messages = ConversationAndSystemMessages()
    conversation_and_system_messages.add_system_message(system_prompt)

    model_responses = []
    model_response_indices = []

    # E               groq.RateLimitError: Error code: 429 - {'error': {'message': 'Rate limit reached for model `llama-3.3-70b-versatile` in organization `org_01hxms56tvftrr5ysv46mybzqc` service tier `on_demand` on tokens per day (TPD): Limit 100000, Used 92762, Requested 12452. Please try again in 1h15m4.753s. Need more tokens? Upgrade to Dev Tier today at https://console.groq.com/settings/billing', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}

#/usr/local/lib/python3.12/dist-packages/groq/_base_client.py:1034: RateLimitError

    try:
        for index, prompt in enumerate(train_english_prompts):
            conversation_and_system_messages.append_message(
                UserMessage(content=prompt["text"]))

            conversation_as_input = \
                conversation_and_system_messages.get_conversation_as_list_of_dicts()

            response = groq_api_wrapper.create_chat_completion(conversation_as_input)
            if response and hasattr(response, "choices") and \
                len(response.choices) > 0:
                assistant_message_content = response.choices[0].message.content
                conversation_and_system_messages.append_message(
                    AssistantMessage(content=assistant_message_content))
                # Uncomment to print the assistant message content
                print(assistant_message_content)
                model_responses.append(response.choices[0].message)
            else:
                print(f"No response received from API call: {response}")
                conversation_and_system_messages.append_general_message(response)

            model_responses.append(response)
            model_response_indices.append(index)
    except Exception as e:
        print(f"Error: {e}")
        print(model_responses)
        if len(model_response_indices) > 0:
            print("model_response_indices[-1]", model_response_indices[-1])

            model_responses_path = Path.cwd() / "model_responses.json"
            JSONFile.save_json(model_responses_path, model_responses)
            conversation_path = Path.cwd() / "conversation.json"
            JSONFile.save_json(
                conversation_path,
                conversation_and_system_messages.get_conversation_as_list_of_dicts())

        else:
            print("model_response_indices is empty")
        raise

    assert len(model_responses) == len(train_prompts)

    model_responses_path = Path.cwd() / "model_responses.json"
    JSONFile.save_json(model_responses_path, model_responses)
    conversation_path = Path.cwd() / "conversation.json"
    JSONFile.save_json(
        conversation_path,
        conversation_and_system_messages.get_conversation_as_list_of_dicts())

    print("\n\n\n Done with model responses")

    for model_response in model_responses:
        try:
            print(model_response.choices[0].message.content)
        except Exception as e:
            print(f"Error: {e}")
            print(model_response)
