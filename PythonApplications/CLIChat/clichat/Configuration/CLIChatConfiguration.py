from pydantic import BaseModel, Field, validator
from typing import Optional

class CLIChatConfiguration(BaseModel):
    cancel_entry: str = Field(default=".cancel")
    exit_entry: str = Field(default=".exit")

    hotkey_cancel: list[str] = Field(default=['c-z'])
    hotkey_exit: list[str] = Field(default=['c-x'])
    hotkey_insert_newline: list[str] = Field(default=['c-i'])
    hotkey_toggle_word_wrap: list[str] = Field(default=['c-w'])
    hotkey_new: list[str] = Field(default=['c-n'])

    temperature: float = Field(default=1.0)
    max_tokens: int = Field(default=8192)

    # TerminalModeDialogs dependencies.
    terminal_DisplayCommandOnMenu: bool = Field(default=False)
    terminal_CommandEntryColor2: str = Field(default="ansigreen")
    terminal_PromptIndicatorColor2: str = Field(default="ansicyan")
    terminal_ResourceLinkColor: str = Field(default="ansiyellow")

    wrap_words: bool = Field(default=True)

    chat_history_path: Optional[str] = Field(default=None)

    @validator('chat_history_path')
    def empty_string_to_none(cls, v):
        if v == "":
            return None
        return v
