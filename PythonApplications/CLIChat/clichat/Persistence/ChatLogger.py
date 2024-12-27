from datetime import datetime
from pathlib import Path
import json

class ChatLogger:
    def __init__(self, log_path: str = "Data/chat_log.txt"):
        self.log_path = Path(log_path)
        self._ensure_log_file_exists()
    
    def _ensure_log_file_exists(self):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.log_path.exists():
            self.log_path.touch()
    
    def log_message(self, role: str, content: str):
        timestamp = datetime.now().timestamp()
        log_entry = {
            "timestamp": timestamp,
            "role": role,
            "content": content
        }
        
        with open(self.log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry) + '\n')
