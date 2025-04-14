from pydantic import BaseModel, Field
from typing import Optional
from pathlib import Path

class CLIChatLocalConfiguration(BaseModel):
    # Command settings
    exit_command: str = Field(default=".exit")
    help_command: str = Field(default=".help")
    
    # UI settings
    user_color: str = Field(default="ansigreen")
    assistant_color: str = Field(default="ansiblue")
    system_color: str = Field(default="ansiyellow")
    info_color: str = Field(default="ansicyan")
    error_color: str = Field(default="ansired")
    
    # File paths
    conversations_dir: Optional[Path] = Field(default=None)
    system_messages_file: Optional[Path] = Field(default=None)
    
    def __init__(self, **data):
        super().__init__(**data)
        
        # Set default paths if not provided
        if not self.conversations_dir:
            self.conversations_dir = Path.home() / \
                ".clichatlocal" / "conversations"
            self.conversations_dir.mkdir(parents=True, exist_ok=True)
            
        if not self.system_messages_file:
            system_dir = Path.home() / ".clichatlocal" / "system_messages"
            system_dir.mkdir(parents=True, exist_ok=True)
            self.system_messages_file = system_dir / "system_messages.json"
