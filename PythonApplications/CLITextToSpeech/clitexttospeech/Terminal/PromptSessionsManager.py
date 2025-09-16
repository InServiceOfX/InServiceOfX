from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import FuzzyCompleter,WordCompleter
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.layout.processors import HighlightMatchingBracketProcessor
from prompt_toolkit.styles import Style
import re

class PromptSessionsManager:
    def __init__(self, app):
        self._app = app

        self._completer = FuzzyCompleter(WordCompleter(
            words=list(self._app._command_handler.commands.keys()),
            meta_dict=self._app._command_handler._command_descriptions,
            pattern=re.compile(r'^\.')
        ))

        self._session = self._create_session()

    def _create_session(self) -> PromptSession:
        style = Style.from_dict({
            # Completion menu styles
            'completion-menu': 'bg:#333333 #ffffff',
            'completion-menu.completion': 'bg:#444444 #ffffff',
            'completion-menu.completion.current': 'bg:#008888 #ffffff',
            'completion-menu.meta': 'bg:#999999 #000000',
            'completion-menu.meta.completion': 'bg:#aaaaaa #000000',
            'completion-menu.meta.completion.current': 'bg:#00aaaa #000000',
            
            # Toolbar style
            'bottom-toolbar': 'bg:#222222 #aaaaaa',
            'bottom-toolbar.text': 'bg:#222222 #ffffff',
        })

        def get_bottom_toolbar():
            """Returns bottom toolbar text"""
            multiline_status = "ON" if getattr(self._session.app, "multiline", False) else "OFF"
            return HTML(
                f'<b>CLITextToSpeech</b> | '
                f'Type .help for commands'
            )

        return PromptSession(
            completer=self._completer,
            bottom_toolbar=get_bottom_toolbar,
            wrap_lines=True,
            style=style,
            auto_suggest=AutoSuggestFromHistory(),
            input_processors=[HighlightMatchingBracketProcessor()],
            complete_in_thread=True,
            complete_while_typing=True,
            # It seems to disallow being able to copy-paste text?
            #mouse_support=True,
            enable_history_search=True,
            search_ignore_case=True,
            enable_system_prompt=True,
            enable_suspend=True,
            enable_open_in_editor=True,
            reserve_space_for_menu=3,
            complete_style="MULTI_COLUMN"
        )

    def prompt(self, prompt_text: str = ">>> ") -> str:
        return self._session.prompt(prompt_text)