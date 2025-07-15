from commonapi.Messages.Messages import Message
from commonapi.Messages.PermanentConversation import PermanentConversation
from TestData.CreateExampleConversation import CreateExampleConversation

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