from clichatlocal.FileIO import JSONFile
from commonapi.Messages import RecordedSystemMessage
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