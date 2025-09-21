from commonapi.Messages import ConversationSystemAndPermanent
from commonapi.Messages.Messages import AssistantMessage, UserMessage
from TestData.CreateExampleConversation import CreateExampleConversation

from pathlib import Path
import json

test_data_path = Path(__file__).parents[2] / "TestData"
test_conversation_path = test_data_path / "test_enable_thinking_true.json"

def load_test_conversation():
    with open(test_conversation_path, "r") as f:
        return json.load(f)

def test_ConversationSystemAndPermanent_inits():
    csp = ConversationSystemAndPermanent()

    assert csp.casm.system_messages_manager._messages_dict == {}
    assert csp.casm.conversation_history.messages == []
    assert csp.get_conversation_as_list_of_dicts() == []
    assert csp.get_permanent_conversation_messages() == []
    assert csp.get_permanent_conversation_message_pairs() == []

def test_ConversationSystemAndPermanent_works():
    conversation = CreateExampleConversation.EXAMPLE_CONVERSATION_0

    conversation_system_and_permanent = ConversationSystemAndPermanent()

    conversation_system_and_permanent.add_system_message(
        conversation[0]["content"])

    for message in conversation[1:]:
        if message["role"] == "user":
            conversation_system_and_permanent.append_message(
                UserMessage(message["content"]))
        elif message["role"] == "assistant":
            conversation_system_and_permanent.append_message(
                AssistantMessage(message["content"]))

    list_of_messages_dict = \
        conversation_system_and_permanent.get_conversation_as_list_of_dicts()

    assert len(list_of_messages_dict) == len(conversation)

    for index, message in enumerate(list_of_messages_dict):
        assert message["role"] == conversation[index]["role"]
        assert message["content"] == conversation[index]["content"]

    assert len(conversation_system_and_permanent.pc.messages) == \
        len(conversation)

    for index, message_pair in enumerate(
        conversation_system_and_permanent.pc.message_pairs):
        i = 2 * index + 1
        assert message_pair.conversation_pair_id == index
        assert message_pair.content_0 == conversation[i]["content"]
        assert message_pair.content_1 == conversation[i+1]["content"]
        assert message_pair.role_0 == conversation[i]["role"]
        assert message_pair.role_1 == conversation[i+1]["role"]

def test_append_message_appends_for_pairs():
    conversation = load_test_conversation()

    csp = ConversationSystemAndPermanent()

    for message in conversation:
        if message["role"] == "user":
            csp.append_message(
                UserMessage(message["content"]))
        elif message["role"] == "assistant":
            csp.append_message(
                AssistantMessage(message["content"]))

    assert len(csp.pc.messages) == len(conversation)
    assert len(csp.pc.message_pairs) == len(conversation) // 2

    for index, message in enumerate(csp.pc.messages):
        assert message.content == conversation[index]["content"]
        assert message.role == conversation[index]["role"]
        assert message.conversation_id == index

    for index, message_pair in enumerate(csp.pc.message_pairs):
        i = 2 * index
        assert message_pair.conversation_pair_id == index
        assert message_pair.content_0 == conversation[i]["content"]
        assert message_pair.content_1 == conversation[i+1]["content"]
        assert message_pair.role_0 == conversation[i]["role"]
        assert message_pair.role_1 == conversation[i+1]["role"]