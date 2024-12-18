from prompt_toolkit.history import FileHistory
from prompt_toolkit import PromptSession

def create_prompt_session(chat_history_path):
    return PromptSession(history=FileHistory(chat_history_path))