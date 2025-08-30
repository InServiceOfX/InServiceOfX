from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
import re

class PromptSessionsManager:
    def __init__(self, app):
        self._app = app

        self._completer = WordCompleter(
            words=list(self._app._command_handler.commands.keys()),
            pattern=re.compile(r'^\.')
        )

        self._session = self._create_session()

    def _create_session(self) -> PromptSession:
        return PromptSession(
            completer=self._completer,
            wrap_lines=True,
            auto_suggest=AutoSuggestFromHistory(),
            complete_while_typing=True,
            complete_style="MULTI_COLUMN"
        )

    def prompt(self, prompt_text: str = ">>> ") -> str:
        return self._session.prompt(prompt_text)