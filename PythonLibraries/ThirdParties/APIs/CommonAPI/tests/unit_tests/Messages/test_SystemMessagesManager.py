from commonapi.Messages import (RecordedSystemMessage, SystemMessagesManager)

def test_SystemMessagesManager_inits():
    system_messages_manager = SystemMessagesManager()

    assert len(system_messages_manager.messages) == 1

    assert system_messages_manager.messages[0].content == \
        RecordedSystemMessage.create_default_message().content

    assert system_messages_manager.messages[0].is_active == True

def create_example_content():
    content_1 = "You are a helpful, knowledgeable assistant."
    content_2 = (
        "You are a math tutor. Always show your reasoning step by step before "
        "giving the final answer.")
    content_3 = \
        "You are a concise summarizer. Respond in no more than two sentences."
    
    hash_1 = RecordedSystemMessage.create_hash(content_1)
    hash_2 = RecordedSystemMessage.create_hash(content_2)
    hash_3 = RecordedSystemMessage.create_hash(content_3)

    content = [content_1, content_2, content_3]
    hashes = [hash_1, hash_2, hash_3]

    return content, hashes

def test_SystemMessagesManager_add_message():
    system_messages_manager = SystemMessagesManager()

    content, hashes = create_example_content()
    system_messages_manager.add_message(content[0])
    system_messages_manager.add_message(content[1])
    system_messages_manager.add_message(content[2])

    assert len(system_messages_manager.messages) == 4

    assert system_messages_manager.messages[0].content == \
        RecordedSystemMessage.create_default_message().content
    assert system_messages_manager.messages[1].content == content[0]
    assert system_messages_manager.messages[2].content == content[1]
    assert system_messages_manager.messages[3].content == content[2]
    
    assert system_messages_manager.get_message_by_hash(hashes[0]).content == \
        content[0]
    assert system_messages_manager.get_message_by_hash(hashes[1]).content == \
        content[1]
    assert system_messages_manager.get_message_by_hash(hashes[2]).content == \
        content[2]
    
    active_messages = system_messages_manager.get_active_messages()    
    assert len(active_messages) == 1
    assert active_messages[0].content == \
        RecordedSystemMessage.create_default_message().content

def test_clear_works():
    system_messages_manager = SystemMessagesManager()

    assert len(system_messages_manager.messages) == 1
    assert system_messages_manager.messages[0].content == \
        RecordedSystemMessage.create_default_message().content

    messages = system_messages_manager.get_all_as_system_message_dicts()
    assert len(messages) == 1
    assert messages[0]["content"] == \
        RecordedSystemMessage.create_default_message().content

    system_messages_manager.clear()

    assert len(system_messages_manager.messages) == 0

    messages = system_messages_manager.get_all_as_system_message_dicts()
    assert len(messages) == 0

    content, hashes = create_example_content()
    system_messages_manager.add_message(content[0])
    system_messages_manager.add_message(content[1])
    system_messages_manager.add_message(content[2])
    assert len(system_messages_manager.messages) == 3
    messages = system_messages_manager.get_all_as_system_message_dicts()
    assert len(messages) == 3
    assert messages[0]["content"] == content[0]
    assert messages[1]["content"] == content[1]
    assert messages[2]["content"] == content[2]

    system_messages_manager.clear()
    assert len(system_messages_manager.messages) == 0
    assert len(system_messages_manager.get_all_as_system_message_dicts()) == 0

def test_SystemMessagesManager_toggle_message():
    system_messages_manager = SystemMessagesManager()

    content, hashes = create_example_content()
    system_messages_manager.add_message(content[0])
    system_messages_manager.add_message(content[1])
    system_messages_manager.add_message(content[2])

    system_messages_manager.toggle_message_to_active(hashes[0])

    active_messages = system_messages_manager.get_active_messages()    
    assert len(active_messages) == 2
    assert active_messages[0].content == \
        RecordedSystemMessage.create_default_message().content
    assert active_messages[1].content == content[0]

    system_messages_manager.toggle_message_to_inactive(
        RecordedSystemMessage.create_hash(active_messages[0].content))
    active_messages = system_messages_manager.get_active_messages()    
    assert len(active_messages) == 1
    assert active_messages[0].content == content[0]

def test_SystemMessagesManager_toggle_messages_from_messages():
    system_messages_manager = SystemMessagesManager()

    content, hashes = create_example_content()
    system_messages_manager.add_message(content[0])
    system_messages_manager.add_message(content[1])
    system_messages_manager.add_message(content[2])

    all_messages = system_messages_manager.messages

    assert system_messages_manager.toggle_message_to_active(
        all_messages[1].hash)
    assert system_messages_manager.toggle_message_to_active(
        all_messages[2].hash)

    active_messages = system_messages_manager.get_active_messages()    
    assert len(active_messages) == 3
    assert active_messages[0].content == \
        RecordedSystemMessage.create_default_message().content
    assert active_messages[1].content == content[0]
    assert active_messages[2].content == content[1]

    assert system_messages_manager.toggle_message_to_inactive(
        all_messages[0].hash)
    assert system_messages_manager.toggle_message_to_inactive(
        all_messages[2].hash)

    active_messages = system_messages_manager.get_active_messages()    
    assert len(active_messages) == 1
    assert active_messages[0].content == content[0]