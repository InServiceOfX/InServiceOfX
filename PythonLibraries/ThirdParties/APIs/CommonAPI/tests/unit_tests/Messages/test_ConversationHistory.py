from commonapi.Messages.Messages import (
    AssistantMessage,
    SystemMessage,
    UserMessage,
)

from commonapi.Messages import RecordedSystemMessage

from commonapi.Messages.ConversationHistory import ConversationHistory

def test_ConversationHistory_init():
    conversation_history = ConversationHistory()
    assert conversation_history.messages == []
    assert conversation_history.content_hashes == []

def test_ConversationHistory_append_message():
    conversation_history = ConversationHistory()
    message = UserMessage(content="Hello, world!")
    conversation_history.append_message(message)
    assert conversation_history.messages == [message]

def test_ConversationHistory_append_message_works_with_multiple_user_messages():
    conversation_history = ConversationHistory()
    message_1 = UserMessage(content="Hello, world!")
    conversation_history.append_message(message_1)
    message_2 = UserMessage(content="How are you?")
    conversation_history.append_message(message_2)
    assert conversation_history.messages == [message_1, message_2]

def test_as_list_of_dicts_works():
    conversation_history = ConversationHistory()
    message_1 = UserMessage(content="Hello, world!")
    conversation_history.append_message(message_1)
    message_2 = UserMessage(content="How are you?")
    conversation_history.append_message(message_2)
    assert conversation_history.as_list_of_dicts() == [
        {"role": "user", "content": "Hello, world!"},
        {"role": "user", "content": "How are you?"}]

def create_test_conversation():
    hash_values = []

    content_0 = "You are a helpful, knowledgeable assistant."
    hash_values.append(ConversationHistory._hash_content(content_0))
    content_1 = "You are a Python programming expert."
    hash_values.append(ConversationHistory._hash_content(content_1))
    content_2 = (
        "You are a concise summarizer. Respond in no more than two sentences.")
    hash_values.append(ConversationHistory._hash_content(content_2))
    content_3 = (
        'What is the sentiment expressed in the following tweet: '
        '"I liked the movie but it was a bit too long."')
    hash_values.append(ConversationHistory._hash_content(content_3))
    content_4 = (
        "Write a detailed paragraph on the causes of the French Revolution.")
    hash_values.append(ConversationHistory._hash_content(content_4))
    content_5 = "Please provide Python code to reverse a linked list."
    hash_values.append(ConversationHistory._hash_content(content_5))

    conversation_history = ConversationHistory()
    message = SystemMessage(content=content_0)
    conversation_history.append_message(message)
    message = SystemMessage(content=content_1)
    conversation_history.append_message(message)
    message = UserMessage(content=content_3)
    conversation_history.append_message(message)
    # Try twice.
    conversation_history.append_message(message)
    message = UserMessage(content=content_4)
    conversation_history.append_message(message)
    message = SystemMessage(content=content_2)
    conversation_history.append_message(message)
    message = UserMessage(content=content_5)
    conversation_history.append_message(message)
    return conversation_history, hash_values

def test_ConversationHistory_get_all_system_messages():
    conversation_history, hash_values = create_test_conversation()
    system_messages = conversation_history.get_all_system_messages()
    assert len(system_messages) == 3
    assert ConversationHistory._hash_content(system_messages[0].content) == \
        hash_values[0]
    assert ConversationHistory._hash_content(system_messages[1].content) == \
        hash_values[1]
    assert ConversationHistory._hash_content(system_messages[2].content) == \
        hash_values[2]

def test_estimate_conversation_content_length():
    conversation_history, _ = create_test_conversation()
    conversation_history_as_list_of_dicts = \
        conversation_history.as_list_of_dicts()
    assert len(conversation_history_as_list_of_dicts) == 7
    assert conversation_history.estimate_conversation_content_length() == \
        len(conversation_history_as_list_of_dicts[0]["content"]) + \
        len(conversation_history_as_list_of_dicts[1]["content"]) + \
        len(conversation_history_as_list_of_dicts[2]["content"]) + \
        len(conversation_history_as_list_of_dicts[3]["content"]) + \
        len(conversation_history_as_list_of_dicts[4]["content"]) + \
        len(conversation_history_as_list_of_dicts[5]["content"]) + \
        len(conversation_history_as_list_of_dicts[6]["content"])

def test_ConversationHistory_clear():
    conversation_history, _ = create_test_conversation()
    conversation_history.clear()
    assert conversation_history.messages == []
    assert conversation_history.content_hashes == []
    assert conversation_history.hash_to_index_reverse_map == {}

def create_example_system_content():
    content_1 = (
        "You are a math tutor. Always show your reasoning step by step before "
        "giving the final answer.")
    content_2 = \
        "You are a concise summarizer. Respond in no more than two sentences."
    content_3 = (
        "You are a fact-checking assistant. Only provide information you are "
        "certain is accurate.")
    content_4 = (
        "You are a troubleshooting assistant. If you do not know the answer, "
        "state so clearly.")

    hash_1 = RecordedSystemMessage.create_hash(content_1)
    hash_2 = RecordedSystemMessage.create_hash(content_2)
    hash_3 = RecordedSystemMessage.create_hash(content_3)
    hash_4 = RecordedSystemMessage.create_hash(content_4)

    content = [content_1, content_2, content_3, content_4]
    hashes = [hash_1, hash_2, hash_3, hash_4]

    return content, hashes

def test_ConversationHistory_and_SystemMessagesManager_hashes_equally_on_system_messages():
    content, hashes = create_example_system_content()
    assert hashes[0] == ConversationHistory._hash_content(content[0])
    assert hashes[1] == ConversationHistory._hash_content(content[1])
    assert hashes[2] == ConversationHistory._hash_content(content[2])
    assert hashes[3] == ConversationHistory._hash_content(content[3])