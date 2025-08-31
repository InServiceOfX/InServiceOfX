from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter, FuzzyCompleter
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
import re

class PromptSessionsManager:
    def __init__(self, app):
        self._app = app

        self._completer = FuzzyCompleter(WordCompleter(
            words=list(self._app._command_handler.commands.keys()),
            meta_dict=self._app._command_handler.get_command_descriptions(),
            pattern=re.compile(r'^\.')
        ))

        self._session = self._create_session()

    def _create_session(self) -> PromptSession:
        style = self._app._terminal_ui.create_prompt_style()

        return PromptSession(
            completer=self._completer,
            style=style,
            wrap_lines=True,
            auto_suggest=AutoSuggestFromHistory(),
            complete_while_typing=True,
            complete_style="MULTI_COLUMN",
            reserve_space_for_menu=3
        )

    def prompt(self, prompt_text: str = ">>> ") -> str:
        return self._session.prompt(prompt_text)