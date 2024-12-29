from pydantic import BaseModel, Field, field_validator
from typing import Optional
from clichat.Utilities.Formatting import empty_string_to_none

class CLIChatConfiguration(BaseModel):
    cancel_entry: str = Field(default=".cancel")
    exit_entry: str = Field(default=".exit")

    hotkey_cancel: list[str] = Field(default=['c-z'])
    hotkey_exit: list[str] = Field(default=['c-x'])
    hotkey_insert_newline: list[str] = Field(default=['c-i'])
    hotkey_toggle_word_wrap: list[str] = Field(default=['c-w'])
    hotkey_new: list[str] = Field(default=['c-n'])

    temperature: float = Field(default=1.0)

    # TerminalModeDialogs dependencies.
    terminal_CommandEntryColor2: str = Field(default="ansigreen")
    terminal_PromptIndicatorColor2: str = Field(default="ansicyan")
    terminal_ResourceLinkColor: str = Field(default="ansiyellow")

    terminal_DialogBackgroundColor: str = Field(default="ansiblue")

    terminal_SystemMessageColor: str = Field(default="ansigray")
    terminal_SystemMessagePrefix: str = Field(default="🤖")

    terminal_ErrorColor: str = Field(default="ansiyellow")

    chat_history_path: Optional[str] = Field(default=None)
    chat_log_path: Optional[str] = Field(default=None)
    system_messages_path: Optional[str] = Field(default=None)

    @field_validator('chat_history_path')
    def chat_history_path_empty_string_to_none(cls, v):
        return empty_string_to_none(v)

    @field_validator('chat_log_path')
    def chat_log_path_empty_string_to_none(cls, v):
        return empty_string_to_none(v)

    @field_validator('system_messages_path')
    def system_messages_path_empty_string_to_none(cls, v):
        return empty_string_to_none(v)
