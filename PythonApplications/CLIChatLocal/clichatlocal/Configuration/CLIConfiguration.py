from pydantic import BaseModel, Field
from typing import Optional
from pathlib import Path

class CLIConfiguration(BaseModel):
    # Command settings
    exit_command: str = Field(default=".exit")
    help_command: str = Field(default=".help")
    
    # UI settings
    user_color: str = Field(default="ansigreen")
    assistant_color: str = Field(default="ansiblue")
    system_color: str = Field(default="ansiyellow")
    info_color: str = Field(default="ansicyan")
    error_color: str = Field(default="ansired")

    file_history_path: Optional[Path] = None

    def __init__(self, is_dev: bool = False, **data):

        super().__init__(**data)

        if "file_history_path" not in data or data["file_history_path"] is None:
            if is_dev:
                self.file_history_path = Path(__file__).parents[1] / \
                    "Configurations" / "file_history.txt"
            else:
                self.file_history_path = Path.home() / ".clichatlocal" / \
                    "file_history.txt"