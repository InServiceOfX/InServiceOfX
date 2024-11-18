from langchain.memory import ChatMessageHistory

class OldStyleChatHistoryStore:
    store = {}

    def __init__(self):
        self.store = {}

    def get_chat_history(self, session_id: str):
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store.get(session_id, [])