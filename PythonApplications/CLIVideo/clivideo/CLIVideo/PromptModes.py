import re
from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.formatted_text import HTML
from enum import Enum, auto
from clivideo.Configuration import CLIVideoConfiguration
from clivideo.CLIVideo.VideoGenerationPrompts import VideoGenerationPrompts

class PromptMode(Enum):
    IMAGE_VIDEO = auto()
    IMAGE_GENERATION = auto()  # For Fal AI operations
    # Add other modes as needed
    # FAL_AI = auto()
    # LUMA_AI = auto()

class PromptModes:
    def __init__(self, configuration: CLIVideoConfiguration):
        self.configuration = configuration
        self.current_mode = PromptMode.IMAGE_VIDEO
        self.kb = KeyBindings()
        
        # Create completers dictionary with compiled regex pattern
        dot_pattern = re.compile(r'^\.')
        self.completers = {
            PromptMode.IMAGE_VIDEO: WordCompleter(
                VideoGenerationPrompts.COMMANDS, 
                pattern=dot_pattern
            )
        }
        
        self.session = self._create_session()
        
        # Register mode switching ONLY for the configured key
        @self.kb.add(self.configuration.mode_switch_key)
        def _(event):
            """Switch between available modes"""
            modes = list(PromptMode)
            current_index = modes.index(self.current_mode)
            self.current_mode = modes[(current_index + 1) % len(modes)]
            # Update session completer
            event.app.current_buffer.completer = self.completers.get(self.current_mode)
            event.app.invalidate()

        # Add Enter key binding for normal input acceptance
        @self.kb.add('enter')
        def _(event):
            """Accept input on Enter"""
            event.current_buffer.validate_and_handle()
        
    def _create_session(self) -> PromptSession:
        def get_toolbar():
            """Returns bottom toolbar text based on current mode"""
            return HTML(
                f'Mode: <b fg="{self.configuration.mode_text_color}">'
                f'{self.current_mode.name}</b> '
                f'({self.configuration.mode_switch_key} to switch)'
            )
            
        return PromptSession(
            completer=self.completers.get(self.current_mode),
            bottom_toolbar=get_toolbar,
            key_bindings=self.kb,
            style=self.configuration.prompt_style,
            wrap_lines=True
        )
        
    async def prompt_async(self, prompt_text: str = ">>> ") -> str:
        return await self.session.prompt_async(prompt_text)