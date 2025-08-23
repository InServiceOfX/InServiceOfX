from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
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
            wrap_lines=True
        )