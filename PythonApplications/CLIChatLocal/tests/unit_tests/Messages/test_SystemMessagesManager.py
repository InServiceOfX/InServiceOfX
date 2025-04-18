from clichatlocal.Messages import SystemMessage, SystemMessagesManager


from pathlib import Path

def test_SystemMessagesManager_inits():
    system_messages_manager = SystemMessagesManager()

    assert system_messages_manager.system_messages_file_path == None

    assert len(system_messages_manager.messages) == 1

    assert system_messages_manager.messages[0].content == \
        SystemMessage.create_default_message().content

    assert system_messages_manager.messages[0].is_active == True
    
def test_SystemMessagesManager_add_message():
    system_messages_manager = SystemMessagesManager()

    content_1 = "You are a helpful, knowledgeable assistant."
    content_2 = (
        "You are a math tutor. Always show your reasoning step by step before "
        "giving the final answer.")
    content_3 = \
        "You are a concise summarizer. Respond in no more than two sentences."
    
    hash_1 = SystemMessage.create_hash(content_1)
    hash_2 = SystemMessage.create_hash(content_2)
    hash_3 = SystemMessage.create_hash(content_3)

    system_messages_manager.add_message(content_1)
    system_messages_manager.add_message(content_2)
    system_messages_manager.add_message(content_3)

    assert len(system_messages_manager.messages) == 4

    assert system_messages_manager.messages[0].content == \
        SystemMessage.create_default_message().content
    assert system_messages_manager.messages[1].content == content_1
    assert system_messages_manager.messages[2].content == content_2
    assert system_messages_manager.messages[3].content == content_3
    
    assert system_messages_manager.get_message_by_hash(hash_1).content == \
        content_1
    assert system_messages_manager.get_message_by_hash(hash_2).content == \
        content_2
    assert system_messages_manager.get_message_by_hash(hash_3).content == \
        content_3
    
    active_messages = system_messages_manager.get_active_messages()    
    assert len(active_messages) == 1
    assert active_messages[0].content == \
        SystemMessage.create_default_message().content
