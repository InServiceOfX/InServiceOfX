# This avoids circular import errors; notice that JSONFile.py is in the same
# directory as this file.
from clichatlocal.FileIO.JSONFile import JSONFile
from commonapi.Messages.RecordedMessages import RecordedMessage, RecordedUserMessage, RecordedAssistantMessage
from pathlib import Path
from typing import List

class ConversationHistoryFileIO:
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
            messages = []
            for msg in data:
                if msg['role'] == 'user':
                    messages.append(RecordedUserMessage(**msg))
                elif msg['role'] == 'assistant':
                    messages.append(RecordedAssistantMessage(**msg))
                elif msg['role'] == 'system':
                    messages.append(RecordedMessage(**msg))
            
            self.messages = messages
            return True
        except (KeyError, TypeError):
            return False

    def save_messages(self, messages: List[RecordedMessage]) -> bool:
        """
        Save messages by appending to an existing file rather than overwriting it.
        Does not create the file if it doesn't exist.
        
        Args:
            messages: List of RecordedMessage objects to save
            
        Returns:
            bool: True if successful, False otherwise
        """
        if self.file_path is None:
            return False
        
        file_path = Path(self.file_path)
        if not file_path.exists():
            # File doesn't exist, don't try to create it
            return False
        
        # Convert messages to JSON-serializable format
        data = [msg.__dict__ for msg in messages]
        
        # Append to the file instead of overwriting
        try:
            # If the file is empty, write as a new JSON array
            if file_path.stat().st_size == 0:
                return JSONFile.save_json(file_path, data)
            
            # Otherwise, read existing content, append new messages, and save
            existing_data = JSONFile.load_json(file_path) or []
            combined_data = existing_data + data
            
            return JSONFile.save_json(file_path, combined_data)
        except Exception as e:
            print(f"Error appending to file: {e}")
            return False
    
    def put_messages_into_permanent_conversation_history(
        self,
        permanent_conversation_history) -> bool:
        if self.messages != None and self.messages != []:
            for _, message in self.messages.items():
                permanent_conversation_history.append_recorded_message(message)
            return True
        return False

    def ensure_file_exists(self, file_path=None) -> bool:
        """
        Ensures that the conversation history file exists, creating it if
        necessary. Optionally updates the file path.
        
        Args:
            file_path: Optional new file path to use
            
        Returns:
            bool: True if file exists or was created successfully, False
            otherwise
        """
        # Update file path if provided
        if file_path is not None:
            self.file_path = file_path
        
        # Check if file path is set
        if self.file_path is None:
            return False
        
        path = Path(self.file_path)
        
        # Check if file already exists
        if path.exists():
            return True
        
        try:
            # Create parent directories if they don't exist
            path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create empty file
            with open(path, 'w') as f:
                f.write('[]')  # Initialize with empty JSON array
            
            return True
        except Exception as e:
            print(f"Error creating conversation history file: {e}")
            return False