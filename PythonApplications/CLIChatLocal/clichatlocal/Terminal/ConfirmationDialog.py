from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.styles import Style
from typing import Optional

class ConfirmationDialog:
    """Simple inline confirmation dialog for yes/no questions."""
    
    def __init__(self, style: Style):
        self.style = style
        self.completer = \
            WordCompleter(['yes', 'no', 'y', 'n'], ignore_case=True)

        self._prompt_session_manager = None

    def setup_with_prompt_session_manager(self, prompt_session_manager):
        self._prompt_session_manager = prompt_session_manager

    async def ask_confirmation(self, question: str, default: str = None) \
        -> bool:
        """
        Ask a yes/no question inline.
        
        Args:
            question: The question to ask
            default: Default answer ('yes' or 'no')
            
        Returns:
            bool: True if yes, False if no
        """
        if default is None:
            default = 'yes'
        # Format the question with default
        prompt_text = \
            f"{question} [{'Y' if default.lower() == 'yes' else 'N'}]: "
        
        try:
            answer = await self._prompt_session_manager.prompt_async(
                prompt_text)
            answer = answer.strip().lower()
            
            # Handle empty input (use default)
            if not answer:
                return default.lower() in ['yes', 'y']

            # Handle various yes/no formats
            if answer in ['yes', 'y', '1', 'true']:
                return True
            elif answer in ['no', 'n', '0', 'false']:
                return False
            else:
                # Invalid input, use default
                return default.lower() in ['yes', 'y']
                
        except KeyboardInterrupt:
            # Ctrl+C pressed, use default
            return default.lower() in ['yes', 'y']
        except Exception:
            # Any other error, use default
            return default.lower() in ['yes', 'y']