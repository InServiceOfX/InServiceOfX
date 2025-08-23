from pathlib import Path
import asyncio

from clichatlocal.Configuration.CLIConfiguration import CLIConfiguration
from clichatlocal.Configuration import ModelList
from clichatlocal.Terminal import (
    TerminalUI,
    PromptSessionManager,
    CommandHandler)

from moretransformers.Applications import LocalLlama3
from moretransformers.Configurations import Configuration, GenerationConfiguration

class CLIChatLocal:
    def __init__(self, application_paths):
        self._model_list = ModelList.from_yaml(model_list_file_path)
        first_model_name = next(iter(self._model_list.models))
        first_model_path = self._model_list.models[first_model_name]
        
        print(f"First model: {first_model_name}")
        print(f"First model path: {first_model_path}")

        self.cli_configuration = CLIConfiguration()
        
        # Initialize components
        self.terminal_ui = TerminalUI(self.cli_configuration)
        
        # Setup command handler
        self.command_handler = CommandHandler(self)
        
        # Create prompt session
        self._psm = PromptSessionManager(
            self.cli_configuration)
    
    async def run_iterative(self):
        """Single iteration of chat interaction"""
        try:
            prompt = await self._psm.session.prompt_async(
                "Chat prompt (or type .help for options): "
            )
            
            if not prompt.strip():
                return True

            # Check if it's a command
            if prompt.strip().startswith('.'):
                continue_running, command_handled = \
                    await self.command_handler.handle_command(prompt)

                # If command wasn't handled, treat as regular user input
                if not command_handled:
                    self.terminal_ui.print_user_message(prompt)
                    # response = self.llama3_engine.generate_from_single_user_content(prompt)
                    # self.terminal_ui.print_assistant_message(response)
                
                return continue_running

            # Generate response
            self.terminal_ui.print_user_message(prompt)
            response = self.llama3_engine.generate_from_single_user_content(
                prompt)
            self.terminal_ui.print_assistant_message(response)

            return True
            
        except KeyboardInterrupt:
            # Handle Ctrl+C exit
            self.terminal_ui.print_info("Saving conversation history...")
            self.terminal_ui.print_info("Goodbye!")
            return False
        except Exception as e:
            self.terminal_ui.print_error(f"Error: {str(e)}")
            return True
    
    def run(self):
        """Main run loop"""
        print("Running CLIChatLocal")
        self.terminal_ui.print_header("Welcome to CLIChatLocal!")
        self.terminal_ui.print_info("Press Ctrl+C to exit.")
        
        async def run_async():
            continue_running = True
            while continue_running:
                continue_running = await self.run_iterative()

        asyncio.run(run_async())
        self.terminal_ui.print_info("Thank you for using CLIChatLocal!")
