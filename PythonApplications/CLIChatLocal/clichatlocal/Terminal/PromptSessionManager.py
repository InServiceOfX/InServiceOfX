import re
from typing import Optional, Dict, List, Any, Callable

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter, FuzzyCompleter
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.filters import Condition
from prompt_toolkit.layout.processors import HighlightMatchingBracketProcessor

class PromptSessionManager:
    """Manages the prompt session with advanced features."""
    
    def __init__(self, cli_configuration):
        self.cli_configuration = cli_configuration
        self.kb = KeyBindings()
        
        # Define available commands with descriptions
        self.commands = {
            ".help": "Show available commands",
            ".exit": "Exit the application",
            ".clear": "Clear conversation history",
            ".system": "Set or view system message",
            ".save": "Save current conversation",
            ".load": "Load a saved conversation",
            ".temp": "Set temperature (0.0-2.0)",
            ".tokens": "Set max tokens for generation",
            ".history": "View conversation history"
        }
        
        # Create completer with command descriptions
        self.completer = FuzzyCompleter(WordCompleter(
            words=list(self.commands.keys()),
            meta_dict=self.commands,
            pattern=re.compile(r'^\.')
        ))
        
        # Setup key bindings
        self._setup_key_bindings()
        
        # Create the prompt session
        self.session = self._create_session()
    
    def _setup_key_bindings(self):
        """Setup key bindings for the prompt session."""
# Add clear screen (Ctrl+L)
        @self.kb.add('c-l')
        def _(event):
            """Clear the screen"""
            event.app.renderer.clear()
    
    def _create_session(self) -> PromptSession:
        """Create and configure the prompt session."""
        
        # Create a style based on configuration
        style = Style.from_dict({
            # User input style
            '': self.cli_configuration.user_color,
            
            # Prompt indicator
            'indicator': self.cli_configuration.info_color,
            
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
        
        # Create history file path
        history_file = self.cli_configuration.conversations_dir / \
            "command_history.txt"
        history_file.parent.mkdir(parents=True, exist_ok=True)
        
        def get_bottom_toolbar():
            """Returns bottom toolbar text"""
            multiline_status = "ON" if getattr(self.session.app, "multiline", False) else "OFF"
            return HTML(
                f'<b>CLIChatLocal</b> | '
                f'<style fg="{self.cli_configuration.info_color}">Multiline: {multiline_status}</style> '
                f'(Ctrl+M to toggle) | Type .help for commands'
            )
        
        # Create and return the session with all available options
        return PromptSession(
            completer=self.completer,
            bottom_toolbar=get_bottom_toolbar,
            key_bindings=self.kb,
            style=style,
            wrap_lines=True,
            history=FileHistory(str(history_file)),
            auto_suggest=AutoSuggestFromHistory(),
            input_processors=[HighlightMatchingBracketProcessor()],
            complete_in_thread=True,
            complete_while_typing=True,
            mouse_support=True,
            enable_history_search=True,
            search_ignore_case=True,
            enable_system_prompt=True,
            enable_suspend=True,
            enable_open_in_editor=True,
            reserve_space_for_menu=3,
            complete_style="MULTI_COLUMN"
        )
    
    async def prompt_async(self, prompt_text: str = ">>> ") -> str:
        """Get input from the user asynchronously."""
        return await self.session.prompt_async(prompt_text)
    
    def prompt(self, prompt_text: str = ">>> ") -> str:
        """Get input from the user synchronously."""
        return self.session.prompt(prompt_text)
