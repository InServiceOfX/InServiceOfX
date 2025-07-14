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
    