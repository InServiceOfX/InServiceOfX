from corecode.FileIO import JSONFile
from commonapi.Messages import (
    RecordedSystemMessage,
    SystemMessagesManager)
from pathlib import Path
from typing import List

class SystemMessagesFileIO:
    def __init__(self, file_path: str | Path | None = None):
        if isinstance(file_path, str):
            file_path = Path(file_path)

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

    def is_file_path_valid(self) -> bool:
        return self.file_path is not None and Path(self.file_path).exists()

    def save_messages(self, messages: List[RecordedSystemMessage]) -> bool:
        if not self.is_file_path_valid():
            return False

        data = [msg.__dict__ for msg in messages]
        return JSONFile.save_json(self.file_path, data)

    def put_messages_into_system_messages_manager(
        self,
        system_messages_manager: SystemMessagesManager) -> bool:
        if self.messages != None and self.messages != []:
            for _, message in self.messages.items():
                system_messages_manager.add_previously_recorded_message(message)
            return True
        return False
