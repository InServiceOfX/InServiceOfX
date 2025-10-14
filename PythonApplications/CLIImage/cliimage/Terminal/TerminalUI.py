from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.shortcuts import print_formatted_text, clear
from prompt_toolkit.styles import Style
from html import escape

class TerminalUI:
    """
    Handles all formatted text output and user interface elements.
    """
    
    def __init__(self):
        self._setup_styles()
    
    def _setup_styles(self):
        self.style = Style.from_dict({
            # Core message types
            'header': 'bold fg:#ffffff bg:#2c3e50',
            'info': 'fg:#3498db',
            'success': 'fg:#27ae60 bold',
            'error': 'fg:#e74c3c bold',
            #'warning': 'fg:#f39c12 bold',
            'processing': 'fg:#9b59b6 italic',
            
            # User interface elements
            # 'user': 'fg:#34495e bold',
            # 'assistant': 'fg:#2980b9',
            # 'system': 'fg:#7f8c8d italic',
            
            # Special formatting
            'goodbye': 'fg:#27ae60 bg:#ecf0f1 bold',
            'help': 'fg:#2c3e50 bg:#ecf0f1',
            'separator': 'fg:#95a5a6',
            #'highlight': 'fg:#e67e22 bold',
        })
    
    def clear_screen(self):
        clear()
    
    def print_header(self, text: str):
        separator = '─' * 60
        print_formatted_text(HTML(f"\n<header>  {text}  </header>"))
        print_formatted_text(HTML(f"<separator>{separator}</separator>\n"))
    
    def print_info(self, message: str):
        print_formatted_text(HTML(f"<info>ℹ {message}</info>"))
    
    def print_success(self, message: str):
        print_formatted_text(HTML(f"<success>✓ {message}</success>"))
    
    def print_error(self, message: str):
        print_formatted_text(HTML(f"<error>✗ {message}</error>"))
    
    def print_warning(self, message: str):
        """
        Print a warning message with orange styling.
        
        Args:
            message: The warning message to display
        """
        print_formatted_text(HTML(f"<warning>⚠ {message}</warning>"))
    
    def print_processing(self, message: str):
        print_formatted_text(HTML(f"<processing>⟳ {message}</processing>"))
    
    def print_goodbye(self):
        print_formatted_text(HTML("\n<goodbye>👋 Goodbye! Thanks for using CLIImage!</goodbye>\n"))

    def print_help(self, help_text: str):
        print_formatted_text(HTML(f"\n<help>📖 Help:</help>"))
        print_formatted_text(HTML(f"<help>{help_text}</help>\n"))
    
    def print_separator(self):
        """Print a visual separator line."""
        separator = '─' * 60
        print_formatted_text(HTML(f"<separator>{separator}</separator>"))
    
    def create_prompt_style(self):
        return Style.from_dict({
            # User input style
            "": "fg:#34495e",
            "indicator": "fg:#27ae60 bold",
            
            # Completion menu styles
            "completion-menu": "bg:#333333 #ffffff",
            "completion-menu.completion": "bg:#444444 #ffffff",
            "completion-menu.completion.current": "bg:#008888 #ffffff",
            "completion-menu.meta": "bg:#999999 #000000",
            "completion-menu.meta.completion": "bg:#aaaaaa #000000",
            "completion-menu.meta.completion.current": "bg:#00aaaa #000000",
        })
