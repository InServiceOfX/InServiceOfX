from clichatlocal.FileIO import JSONFile
from commonapi.Messages import RecordedSystemMessage, SystemMessagesManager
from commonapi.Messages.RecordedMessages import RecordedMessage
from pathlib import Path
from typing import List

class SystemMessagesFileIO:
    def __init__(self, file_path = None):
        self.file_path = file_path
        self.messages = None

    def load_messages(self) -> bool:
        if self.file_path is None or not Path(self.file_path).exists():
            return False

        data = JSONFile.load_json(self.file_path)
        if not data:
            self.messages = []
            return False

        try:
            messages = [RecordedSystemMessage(**msg) for msg in data]
            self.messages = {msg.hash: msg for msg in messages}
            return True
        except (KeyError, TypeError):
            return False

    def save_messages(self, messages: List[RecordedSystemMessage]) -> bool:
        if self.file_path is None or not Path(self.file_path).exists():
            return False

        data = [msg.__dict__ for msg in messages]
        return JSONFile.save_json(self.file_path, data)
    
    def is_file_path_valid(self) -> bool:
        return self.file_path is not None and Path(self.file_path).exists()

    def put_messages_into_system_messages_manager(
        self,
        system_messages_manager: SystemMessagesManager) -> bool:
        if self.messages != None and self.messages != []:
            for _, message in self.messages.items():
                system_messages_manager.add_previously_recorded_message(message)
            return True
        return False

    def put_messages_into_permanent_conversation_history(
        self,
        permanent_conversation_history) -> bool:
        if self.messages != None and self.messages != []:
            for _, message in self.messages.items():
                recorded_message = RecordedMessage(
                    message.content,
                    message.timestamp,
                    message.hash,
                    "system")
                permanent_conversation_history.append_recorded_message(
                    recorded_message)
            return True
        return False