from dataclasses import dataclass
from typing import List, Dict, Optional

@dataclass
class Message:
    role: str
    content: str
    timestamp: Optional[int] = None

    def to_dict(self):
        """Convert Message to a dictionary, excluding None values"""
        d = {"role": self.role, "content": self.content}
        if self.timestamp is not None:
            d["timestamp"] = self.timestamp
        return d

class MessagesManager:
    def __init__(self):
        self.messages: List[Dict[str, str]] = []
        self.reset_messages()
    
    def reset_messages(self):
        self.messages = [
            create_system_message(msg.content) 
            for msg in self.system_messages_manager.get_active_messages()]
    
    def append_message(self, message: Message):
        self.messages.append(message.to_dict())
    
    def remove_last_user_message(self):
        if self.messages and self.messages[-1]["role"] == "user":
            self.messages.pop()
