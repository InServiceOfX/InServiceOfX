from pathlib import Path
import asyncio

# from prompt_toolkit.formatted_text import HTML
# from prompt_toolkit.shortcuts import print_formatted_text, clear

from clichatlocal.Configuration.CLIConfiguration import CLIConfiguration
from clichatlocal.Terminal import (
    TerminalUI,
    PromptSessionManager,
    CommandHandler)

from moretransformers.Applications import LocalLlama3
from moretransformers.Configurations import Configuration, GenerationConfiguration

class CLIChatLocal:
    def __init__(
        self,
        llama3_configuration_path: Path,
        llama3_generation_configuration_path: Path
    ):
        self.llama3_configuration = Configuration.from_yaml(
            llama3_configuration_path)
        self.llama3_generation_configuration = \
            GenerationConfiguration.from_yaml(
                llama3_generation_configuration_path)
        self.llama3_engine = LocalLlama3(
            self.llama3_configuration,
            self.llama3_generation_configuration)

        self.cli_configuration = CLIConfiguration()

        # # Setup paths for model configurations
        # self.setup_paths()
        
        # # Initialize components
        self.terminal_ui = TerminalUI(self.cli_configuration)
        # self.conversation_manager = ConversationManager(self.config)
        
        # # Setup command handler
        self.command_handler = CommandHandler(self)
        
        # Create prompt session
        self.prompt_session_manager = PromptSessionManager(
            self.cli_configuration)

        # Track history
        self.prompt_history = []
        self.last_prompt = None
    
    async def run_iterative(self):
        """Single iteration of chat interaction"""
        try:
            prompt = await self.prompt_session_manager.session.prompt_async(
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
                    response = self.llama3_engine.generate_from_single_user_content(prompt)
                    self.terminal_ui.print_assistant_message(response)
                
                return continue_running

            # Process regular prompt
            self.prompt_history.append(prompt)
            self.last_prompt = prompt
            
            # Generate response
            # self.terminal_ui.print_user_message(prompt)
            response = self.llama3_engine.generate_from_single_user_content(
                prompt)
            self.terminal_ui.print_assistant_message(response)
            
            return True
            
        except KeyboardInterrupt:
            # Handle Ctrl+C exit
            self.terminal_ui.print_info("Saving conversation history...")
            #self.conversation_manager.save_current_conversation(self.llm.conversation_history)
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
