from pathlib import Path
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit import print_formatted_text

from clitexttospeech.Configuration import CLIConfiguration
from clitexttospeech.Terminal.CommandHandler import CommandHandler
from clitexttospeech.Terminal.PromptSessionsManager import PromptSessionsManager

class CLITextToSpeech:
    def __init__(self, configuration_file_path: Path):
        self.configuration_file_path = configuration_file_path
        self._cli_configuration = CLIConfiguration.from_yaml(configuration_file_path)
        
        # Setup command handler
        self._command_handler = CommandHandler(self)
        
        self._psm = PromptSessionsManager(self)

    def run_iterative(self):
        try:
            prompt = self._psm._session.prompt(
                "Text-to-Speech prompt (or type .help for options): "
            )
            
            if not prompt.strip():
                return True
                
            if prompt.startswith('.'):
                continue_running, command_handled = \
                    self._command_handler.handle_command(prompt)

                # If command wasn't handled, treat as regular user input
                if not command_handled:
                    print(f"Processing text: {prompt}")
                
                return continue_running
            
            # Treat as text input for generation
            print(f"Processing text: {prompt}")
            return True
            
        except KeyboardInterrupt:
            print_formatted_text(HTML("\n<ansigreen>Goodbye!</ansigreen>"))
            return False
        except Exception as e:
            print_formatted_text(
                HTML(f"\n<ansired>Error: {str(e)}</ansired>\n"))
            return True
    
    def run(self):
        """Main run loop"""
        print("Welcome to CLITextToSpeech! Press Ctrl+C to exit.\n")

        continue_running = True
        while continue_running:
            continue_running = self.run_iterative()

        print("\nThank you for using CLITextToSpeech!")