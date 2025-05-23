from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.shortcuts import print_formatted_text, clear
from prompt_toolkit.styles import Style
from html import escape

class TerminalUI:
    def __init__(self, config):
        self.config = config
        
        # Define styles
        self.style = Style.from_dict({
            'user': config.user_color,
            'assistant': config.assistant_color,
            'system': config.system_color,
            'info': config.info_color,
            'error': config.error_color,
            'header': 'bold',
        })
    
    def clear_screen(self):
        """Clear the terminal screen"""
        clear()
    
    def print_header(self, text):
        """Print a formatted header"""
        print_formatted_text(HTML(f"\n<header>{'=' * 50}</header>"))
        print_formatted_text(HTML(f"<header>{text.center(50)}</header>"))
        print_formatted_text(HTML(f"<header>{'=' * 50}</header>\n"))
    
    def print_user_message(self, message):
        """
        Print a user message with proper formatting.
        
        Args:
            message: The message to print
        """
        # Escape any HTML special characters in the message
        escaped_message = escape(message)
        
        formatted_message = \
            f"<{self.config.user_color}>User: {escaped_message}</{self.config.user_color}>"
        print_formatted_text(HTML(formatted_message))
    
    def print_assistant_message(self, message):
        """Print an assistant message"""
        # Escape any HTML-like content in the message
        safe_message = escape(message)
        print_formatted_text(HTML(f"\n<assistant>Assistant:</assistant> {safe_message}\n"))
    
    def print_system_message(self, message):
        """Print a system message"""
        print_formatted_text(HTML(f"\n<system>System:</system> {message}\n"))
    
    def print_info(self, message):
        """Print an informational message"""
        print_formatted_text(HTML(f"<info>{message}</info>"))
    
    def print_error(self, message):
        """Print an error message"""
        print_formatted_text(HTML(f"<error>Error: {message}</error>"))

    def create_prompt_style(self):
        return Style.from_dict({
            "": self.config.terminal_CommandEntryColor2,
            "indicator": self.config.terminal_PromptIndicatorColor2,
        })
            