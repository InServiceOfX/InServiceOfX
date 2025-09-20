from commonapi.Messages.Messages import Message
from commonapi.Messages.PermanentConversation import PermanentConversation
from TestData.CreateExampleConversation import CreateExampleConversation

from pathlib import Path
import json

test_data_path = Path(__file__).parents[2] / "TestData"
test_conversation_path = test_data_path / "test_enable_thinking_true.json"

def load_test_conversation():
    with open(test_conversation_path, "r") as f:
        return json.load(f)

def test_PermanentConversation_works():
    conversation = CreateExampleConversation.EXAMPLE_CONVERSATION_0

    permanent_conversation = PermanentConversation()
    for message in conversation:
        permanent_conversation.add_message_as_content(
            content=message["content"],
            role=message["role"]
            )

    assert len(permanent_conversation.messages) == 7

    for index, message in enumerate(permanent_conversation.messages):
        assert message.conversation_id == index
        assert message.content == conversation[index]["content"]
        assert message.hash == Message._hash_content(message.content)
        assert message.role == conversation[index]["role"]
        assert message.embedding is None

    assert len(permanent_conversation.content_hashes) == len(conversation)

def test_duplicate_messages_can_be_added():
    conversation = CreateExampleConversation.EXAMPLE_CONVERSATION_0

    permanent_conversation = PermanentConversation()
    for message in conversation:
        permanent_conversation.add_message_as_content(
            content=message["content"],
            role=message["role"]
            )

    for _ in range(2):
        permanent_conversation.add_message_as_content(
            content=conversation[5]["content"],
            role=conversation[5]["role"]
            )
        permanent_conversation.add_message_as_content(
            content=conversation[6]["content"],
            role=conversation[6]["role"]
            )

    assert len(permanent_conversation.messages) == 11

    for i in range(7, 11):
        j = (i - 1) % 2 + 5

        assert permanent_conversation.messages[i].content == \
            conversation[j]["content"]
        assert permanent_conversation.messages[i].role == \
            conversation[j]["role"]
        assert permanent_conversation.messages[i].hash == \
            Message._hash_content(conversation[j]["content"])
        assert permanent_conversation.messages[i].conversation_id == i
        assert permanent_conversation.messages[i].embedding is None

def test_PermanentConversation_can_load_example_conversation():
    conversation = load_test_conversation()

    permanent_conversation = PermanentConversation()
    is_next_message_assistant = False
    last_message = None
    for message in conversation:
        permanent_conversation.add_message_as_content(
            content=message["content"],
            role=message["role"]
            )

        if message["role"] == "user":
            is_next_message_assistant = True
            last_message = message
        elif message["role"] == "assistant":
            if is_next_message_assistant and last_message is not None:
                permanent_conversation.add_message_pair_as_content(
                    content_0=last_message["content"],
                    content_1=message["content"],
                    role_0=last_message["role"],
                    role_1=message["role"],
                    embedding=None
                    )
                last_message = None
                is_next_message_assistant = False

    assert permanent_conversation._counter == 16
    assert permanent_conversation._message_pair_counter == 8
    for index, message in enumerate(permanent_conversation.messages):
        assert message.content == conversation[index]["content"]
        assert message.role == conversation[index]["role"]
        assert message.hash == Message._hash_content(message.content)
        assert message.conversation_id == index
        assert message.embedding is None

    message_pair = []
    example_message_pairs = []
    for index, message in enumerate(conversation):
        if index % 2 == 0:
            message_pair.append(message)
        else:
            message_pair.append(message)
            example_message_pairs.append(message_pair)
            message_pair = []

    for message_pair in example_message_pairs:
        assert message_pair[0]["role"] == "user"
        assert message_pair[1]["role"] == "assistant"

    for index, message_pair in enumerate(permanent_conversation.message_pairs):
        assert message_pair.content_0 == \
            example_message_pairs[index][0]["content"]
        assert message_pair.content_1 == \
            example_message_pairs[index][1]["content"]
        assert message_pair.role_0 == example_message_pairs[index][0]["role"]
        assert message_pair.role_1 == example_message_pairs[index][1]["role"]
        assert message_pair.hash == Message._hash_content(
            message_pair.content_0 + message_pair.content_1)
        assert message_pair.conversation_pair_id == index
        assert message_pair.embedding is None

