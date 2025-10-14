from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.shortcuts import print_formatted_text, clear
from prompt_toolkit.styles import Style
from html import escape

from typing import Any

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
            'help': config.info_color,
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

    def print_help(self, help_text: str):
        print_formatted_text(HTML(f"\n<help>📖 Help:</help>"))
        print_formatted_text(HTML(f"<help>{help_text}</help>\n"))

    def create_prompt_style(self):
        return Style.from_dict({
            "": self.config.terminal_CommandEntryColor2,
            "indicator": self.config.terminal_PromptIndicatorColor2,
        })

    def _print_conversation_history(self, messages: list[dict[str, Any]]):
        """Print conversation history in a compact format."""
        if not messages:
            self.print_info("No conversation history available.")
            return
    
        # Header
        print_formatted_text(
            HTML(
                f"\n<header>📜 Last {len(messages)} Messages:</header>"))
        print_formatted_text(HTML(f"<header>{'=' * 60}</header>"))
    
        for i, message in enumerate(messages, 1):
            role = message.get('role', 'unknown')
            content = message.get('content', '')
            
            # Truncate content to fit in ~100 characters
            max_content_length = 100
            if len(content) > max_content_length:
                content = content[:max_content_length-3] + "..."

            # Escape HTML
            safe_content = escape(content)
            
            # Format based on role
            if role == 'user':
                formatted_line = f"<user>{i:2d}. 👤 {safe_content}</user>"
            elif role == 'assistant':
                formatted_line = \
                    f"<assistant>{i:2d}. 🤖 {safe_content}</assistant>"
            elif role == 'system':
                formatted_line = f"<system>{i:2d}. ⚙️  {safe_content}</system>"
            else:
                formatted_line = f"<info>{i:2d}. ❓ {safe_content}</info>"
            
            print_formatted_text(HTML(formatted_line))
        
        print_formatted_text(HTML(f"<header>{'=' * 60}</header>\n"))