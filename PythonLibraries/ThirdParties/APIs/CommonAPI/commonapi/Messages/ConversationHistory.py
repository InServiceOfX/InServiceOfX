from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
import hashlib
from commonapi.Messages.Messages import Message, SystemMessage, AssistantMessage

@dataclass
class ConversationHistory:
    """Stores an ordered history of messages with optional content hashing"""
    messages: List[Message] = None
    content_hashes: List[str] = None
    hash_to_index_reverse_map: Dict[str, int] = None

    def __post_init__(self):
        self.messages = self.messages or []
        self.content_hashes = self.content_hashes or []
        self.hash_to_index_reverse_map = self.hash_to_index_reverse_map or {}

    @staticmethod
    def _hash_content(content: str) -> str:
        """Generate SHA256 hash of message content"""
        return hashlib.sha256(content.encode()).hexdigest()
    
    def append_message(self, message: Message) -> None:
        self.messages.append(message)
        try:
            self.content_hashes.append(self._hash_content(message.content))
            self.hash_to_index_reverse_map[self.content_hashes[-1]] = \
                len(self.content_hashes) - 1
        except AttributeError as err:
            print(f"Error hashing message content: {err}")
            print("type(message): ", type(message))
            print("message: ", message)

    def append_general_message(self, message: Any) -> None:
        """Because the type returned from an API call could be anything
        proprietary, at least handle the case that the type has attribute "role"
        and handle if "role" is "assistant". If there are no "tool_calls" then
        use our defined AssistantMessage type.        
        """
        # Deal with this case:
        # ChatCompletionMessage(content='The result of the mathematical expression 25 * 10 + 10 is 260.', role='assistant', executed_tools=None, function_call=None, reasoning=None, tool_calls=None)
        if hasattr(message, "role") and \
            message.role == "assistant" and \
            hasattr(message, "tool_calls") and \
            (message.tool_calls is None or len(message.tool_calls) == 0):
            self.append_message(AssistantMessage(content=message.content))
        else:
            self.messages.append(message)
            if isinstance(message, dict) and "content" in message:
                self.content_hashes.append(self._hash_content(message["content"]))
                self.hash_to_index_reverse_map[self.content_hashes[-1]] = \
                    len(self.content_hashes) - 1
            # We need to handle the case where the message is of type
            # groq.types.chat.chat_completion_message.ChatCompletionMessage
            # where the content can be None or a string. Consider also this specific
            # example:
            # ChatCompletionMessage(content=None, role='assistant', executed_tools=None, function_call=None, reasoning=None, tool_calls=[ChatCompletionMessageToolCall(id='call_b16f', function=Function(arguments='{"expression": "25 * 10 + 10"}', name='calculate'), type='function')])
            elif hasattr(message, "content"):
                if message.content is not None:
                    self.content_hashes.append(self._hash_content(message.content))
                    self.hash_to_index_reverse_map[self.content_hashes[-1]] = \
                        len(self.content_hashes) - 1
            else:
                try:
                    self.content_hashes.append(self._hash_content(message))
                    self.hash_to_index_reverse_map[self.content_hashes[-1]] = \
                        len(self.content_hashes) - 1
                except Exception as err:
                    print(f"Error hashing message content: {err}")
                    print("type(message): ", type(message))
                    print("message: ", message)

    def delete_message_by_hash(self, hash: str) -> None:
        if hash in self.hash_to_index_reverse_map:
            index = self.hash_to_index_reverse_map[hash]
            del self.hash_to_index_reverse_map[hash]
            self.messages.pop(index)
            self.content_hashes.pop(index)
        else:
            print(f"Hash {hash} not found in hash_to_index_reverse_map")

    def pop_messages(self, count: int = 1, from_start: bool = False) \
        -> List[Message]:
        """Remove and return messages from either end of history"""
        if not self.messages or count <= 0:
            return []

        if from_start:
            removed_messages = self.messages[:count]
            self.messages = self.messages[count:]
            self.content_hashes = self.content_hashes[count:]
            for hash in removed_messages:
                del self.hash_to_index_reverse_map[hash]
        else:
            removed_messages = self.messages[-count:]
            self.messages = self.messages[:-count]
            self.content_hashes = self.content_hashes[:-count]
            for hash in removed_messages:
                del self.hash_to_index_reverse_map[hash]
            
        return removed_messages
    
    def get_conversation_text(self) -> Tuple[str, int, int]:
        """
        Get conversation as formatted string with stats
        Returns:
            Tuple of (conversation_text, char_count, word_count)
        """
        if not self.messages:
            return "", 0, 0
            
        conversation_parts = []
        for message in self.messages:
            conversation_parts.append(message.to_string(message))
            
        conversation_text = "\n".join(conversation_parts)
        char_count = len(conversation_text)
        word_count = len(conversation_text.split())
        
        return conversation_text, char_count, word_count

    def as_list_of_dicts(self) -> List[Dict[str, Any]]:
        return [
            message.to_dict() if isinstance(message, Message) 
            else message if isinstance(message, dict) 
            else message for message in self.messages]

    def clear(self) -> None:
        self.messages.clear()
        self.content_hashes.clear()
        self.hash_to_index_reverse_map.clear()

    def get_all_system_messages(self) -> List[SystemMessage]:
        return [message for message in self.messages \
            if isinstance(message, SystemMessage)]

    def is_message_in_conversation_history_by_hash(self, hash: str) -> bool:
        return hash in self.hash_to_index_reverse_map
