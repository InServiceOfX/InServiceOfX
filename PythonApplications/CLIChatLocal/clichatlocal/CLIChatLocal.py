import asyncio

from clichatlocal.Configuration.CLIConfiguration import CLIConfiguration
from clichatlocal.Core import (
    ModelConversationAndToolsManager,
    ProcessConfigurations)
from clichatlocal.Core.Databases import PostgreSQLResource
from clichatlocal.Core.RAG.PermanentConversation import PostgreSQLAndEmbedding
from clichatlocal.Messages import SystemMessagesDialogHandler
from clichatlocal.Terminal import (
    CommandHandler,
    PromptSessionManager,
    TerminalUI)

class CLIChatLocal:
    def __init__(self, application_paths):
        self._application_paths = application_paths

        self.cli_configuration = CLIConfiguration()
        self._terminal_ui = TerminalUI(self.cli_configuration)
        self._process_configurations = ProcessConfigurations(
            self._application_paths,
            self._terminal_ui)
        self._process_configurations.process_configurations()

        self._mcatm = ModelConversationAndToolsManager(self)
        self._mcatm.load_configurations_and_model()
            
        # Setup command handler. This is before the PromptSessionManager, since
        # the PromptSessionManager needs to know the command names.
        self._command_handler = CommandHandler(self)

        # Create prompt session. This is done after CommandHandler, since
        # CommandHandler supplies the command names to the prompt session.
        self._psm = PromptSessionManager(self, self.cli_configuration)

        self._command_handler._setup_confirmation_dialog(self._psm)

        self._system_messages_dialog_handler = \
            SystemMessagesDialogHandler(self, self.cli_configuration)

        self._pgsqlr = PostgreSQLResource(self._process_configurations)
        self._pgsql_and_embedding = None

    async def setup_postgresql_resource_and_embedding(self):
        await self._pgsqlr.load_configuration_and_create_pool()

        self._pgsql_and_embedding = PostgreSQLAndEmbedding(
            self._pgsqlr.get_connection("PermanentConversation"),
            self._process_configurations
        )
        await self._pgsql_and_embedding.create_tables()
        self._pgsql_and_embedding.setup_embedding_model()
        self._pgsql_and_embedding.create_EmbedPermanentConversation(
            self._mcatm._csp.pc
        )

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
                    self._terminal_ui.print_user_message(prompt)
                    # response = self.llama3_engine.generate_from_single_user_content(prompt)
                    # self._terminal_ui.print_assistant_message(response)

                return continue_running

            # Generate response
            # Commented out because we may not want to re-print the user's
            # message.
            #self._terminal_ui.print_user_message(prompt)
            response = self._mcatm.respond_to_user_message(prompt)
            self._terminal_ui.print_assistant_message(response)

            return True

        except KeyboardInterrupt:
            self._terminal_ui.print_info("Saving conversation history...")
            self._terminal_ui.print_info("Goodbye!")
            return False
        except Exception as e:
            self._terminal_ui.print_error(f"Error: {str(e)}")
            return True

    async def run_iterative_async(self):
        """Single iteration of chat interaction"""
        try:
            # This has to be async because otherwise, you'll get this error:
            # Error: Error: asyncio.run() cannot be called from a running event loop
            # This is possibly because prompt_toolkit itself is also running
            # asyncio.
            prompt = await self._psm.prompt_async(
                "Chat prompt (or type .help for options): "
            )

            if not prompt.strip():
                return True

            # Check if it's a command
            if prompt.strip().startswith('.'):
                if self._command_handler.is_command_async(prompt):
                    continue_running, command_handled = \
                        await self._command_handler.handle_command_async(prompt)
                else:
                    continue_running, command_handled = \
                        self._command_handler.handle_command(prompt)

                # If command wasn't handled, treat as regular user input
                if not command_handled:
                    self._terminal_ui.print_user_message(prompt)
                    # response = self.llama3_engine.generate_from_single_user_content(prompt)
                    # self._terminal_ui.print_assistant_message(response)
                
                return continue_running

            # Generate response
            self._terminal_ui.print_user_message(prompt)
            response = self._mcatm.respond_to_user_message(prompt)
            self._terminal_ui.print_assistant_message(response)

            return True
            
        except KeyboardInterrupt:
            # Handle Ctrl+C exit
            self._terminal_ui.print_info("Saving conversation history...")
            self._terminal_ui.print_info("Goodbye!")
            return False
        except Exception as e:
            self._terminal_ui.print_error(f"Error: {str(e)}")
            return True
    
    async def run_async(self):
        """Main run loop"""
        print("Running CLIChatLocal")
        self._terminal_ui.print_header("Welcome to CLIChatLocal!")
        self._terminal_ui.print_info("Press Ctrl+C to exit.")

        try:
            while True:
                should_continue = await self.run_iterative_async()
                if not should_continue:
                    break
        except KeyboardInterrupt:
            self._terminal_ui.print_info("Goodbye!")
        except Exception as e:
            self._terminal_ui.print_error(f"Error: {str(e)}")
        
        self._terminal_ui.print_info("Thank you for using CLIChatLocal!")

    def run(self):
        print("Running CLIChatLocal")
        self._terminal_ui.print_header("Welcome to CLIChatLocal!")
        self._terminal_ui.print_info("Press Ctrl+C to exit.")

        continue_running = True
        while continue_running:
            continue_running = self.run_iterative()

        self._terminal_ui.print_info("Thank you for using CLIChatLocal!")