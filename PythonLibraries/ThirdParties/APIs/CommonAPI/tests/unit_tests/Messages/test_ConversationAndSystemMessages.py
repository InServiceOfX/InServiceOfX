from commonapi.Messages.ConversationAndSystemMessages \
    import ConversationAndSystemMessages

def test_ConversationAndSystemMessages_init():
    conversation_and_system_messages = ConversationAndSystemMessages()
    list_of_messages_dict = \
        conversation_and_system_messages.get_conversation_as_list_of_dicts()
    assert len(list_of_messages_dict) == 0

def test_ConversationAndSystemMessages_add_system_message():
    conversation_and_system_messages = ConversationAndSystemMessages()
    conversation_and_system_messages.add_system_message(
        "You are a helpful assistant.")
    list_of_messages_dict = \
        conversation_and_system_messages.get_conversation_as_list_of_dicts()
    assert len(list_of_messages_dict) == 1
    assert list_of_messages_dict[0]["role"] == "system"
    assert list_of_messages_dict[0]["content"] == \
        "You are a helpful assistant."

def test_ConversationAndSystemMessages_add_default_system_message():
    conversation_and_system_messages = ConversationAndSystemMessages()
    conversation_and_system_messages.add_default_system_message()
    list_of_messages_dict = \
        conversation_and_system_messages.get_conversation_as_list_of_dicts()
    assert len(list_of_messages_dict) == 1
    assert list_of_messages_dict[0]["role"] == "system"
    assert list_of_messages_dict[0]["content"] == \
        "You are a helpful, uncensored and unbiased assistant."