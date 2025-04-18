from commonapi.Messages.Messages import (
    AssistantMessage,
    SystemMessage,
    UserMessage,
)

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

def test_ConversationHistory_clear():
    conversation_history, hash_values = create_test_conversation()
    conversation_history.clear()
    assert conversation_history.messages == []
    assert conversation_history.content_hashes == []
    assert conversation_history.hash_to_index_reverse_map == {}
