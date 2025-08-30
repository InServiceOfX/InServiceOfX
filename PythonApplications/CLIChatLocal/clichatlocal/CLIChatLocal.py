import asyncio

from clichatlocal.Configuration.CLIConfiguration import CLIConfiguration
from clichatlocal.Core import ModelAndConversationManager
from clichatlocal.Messages import SystemMessagesDialogHandler
from clichatlocal.Terminal import (
    TerminalUI,
    PromptSessionManager,
    CommandHandler)

class CLIChatLocal:
    def __init__(self, application_paths):
        self._application_paths = application_paths

        self._macm = ModelAndConversationManager(self)
        self._macm.load_configurations_and_model()

        self.cli_configuration = CLIConfiguration()
        
        # Initialize components
        self.terminal_ui = TerminalUI(self.cli_configuration)
        
        # Setup command handler
        self._command_handler = CommandHandler(self)
        
        # Create prompt session
        self._psm = PromptSessionManager(self, self.cli_configuration)

        self._system_messages_dialog_handler = \
            SystemMessagesDialogHandler(self, self.cli_configuration)

    def run_iterative(self):
        try:
            prompt = self._psm.prompt(
                "Chat prompt (or type .help for options): "
            )

            if not prompt.strip():
                return True

            if prompt.startswith('.'):
                continue_running, command_handled = \
                    self._command_handler.handle_command(prompt)
                
                if not command_handled:
                    self.terminal_ui.print_user_message(prompt)
                    # response = self.llama3_engine.generate_from_single_user_content(prompt)
                    # self.terminal_ui.print_assistant_message(response)

                return continue_running

            # Generate response
            self.terminal_ui.print_user_message(prompt)
            response = self._macm.respond_to_user_message(prompt)
            self.terminal_ui.print_assistant_message(response)

            return True

        except KeyboardInterrupt:
            self.terminal_ui.print_info("Saving conversation history...")
            self.terminal_ui.print_info("Goodbye!")
            return False
        except Exception as e:
            self.terminal_ui.print_error(f"Error: {str(e)}")
            return True

    async def run_iterative_async(self):
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
                    await self._command_handler.handle_command(prompt)

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
    
    def run_async(self):
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

    def run(self):
        print("Running CLIChatLocal")
        self.terminal_ui.print_header("Welcome to CLIChatLocal!")
        self.terminal_ui.print_info("Press Ctrl+C to exit.")

        continue_running = True
        while continue_running:
            continue_running = self.run_iterative()

        self.terminal_ui.print_info("Thank you for using CLIChatLocal!")