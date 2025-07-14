from commonapi.Messages import ConversationSystemAndPermanent
from commonapi.Messages.Messages import AssistantMessage, UserMessage
from TestData.CreateExampleConversation import CreateExampleConversation

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

    assert len(conversation_system_and_permanent.permanent_conversation.messages) == \
        len(conversation)

    for index, message_pair in enumerate(
        conversation_system_and_permanent.permanent_conversation.message_pairs):
        i = 2 * index + 1
        assert message_pair.conversation_pair_id == index
        assert message_pair.content_0 == conversation[i]["content"]
        assert message_pair.content_1 == conversation[i+1]["content"]
        assert message_pair.role_0 == conversation[i]["role"]
        assert message_pair.role_1 == conversation[i+1]["role"]
