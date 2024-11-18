# This is being deprecated, both in the warning and in the documentation
# https://python.langchain.com/docs/how_to/trim_messages/#using-with-chatmessagehistory
# in favor of say InMemoryChatMessageHistory.
from langchain.memory import ChatMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

def test_ChatMessageHistory_inits():
    history = ChatMessageHistory()
    assert history is not None
    assert isinstance(history, ChatMessageHistory)

def test_InMemoryChatMessageHistory_inits():
    """
    See libs/core/langchain_core/chat_history.py for implementation
    class InMemoryChatMessageHistory(BaseChatmessageHistory, Basemodel)
    """
    history = InMemoryChatMessageHistory()
    assert history is not None
    assert isinstance(history, InMemoryChatMessageHistory)

def test_InMemoryChatMessageHistory_adds_messages_in_bulk():
    """
    See code comments for class BaseChatMessageHistory(ABC) for chat_history.py
    for def add_user_message(..), def add_ai_message(..); "Code should favor the
    bulk add_messages interface instead to save on round-trips to the underlying
    persistence layer."
    """
    history = InMemoryChatMessageHistory()
    # https://python.langchain.com/docs/versions/migrating_memory/chat_history/#chatmessagehistory

    input_message = HumanMessage(content="hi! I'm bob")
    output_message = AIMessage(content="hi! I'm alice")
    input_message_2 = HumanMessage(content="what was my name?")
    output_message_2 = AIMessage(content="bob")

    history.add_messages(
        [input_message, output_message, input_message_2, output_message_2])
    assert history.messages == \
        [input_message, output_message, input_message_2, output_message_2]