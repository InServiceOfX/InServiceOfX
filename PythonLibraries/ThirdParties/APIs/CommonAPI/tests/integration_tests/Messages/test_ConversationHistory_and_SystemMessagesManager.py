from commonapi.Messages import RecordedSystemMessage
from commonapi.Messages.ConversationHistory import ConversationHistory

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
