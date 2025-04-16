from pathlib import Path
# import sys
import asyncio

# from prompt_toolkit import PromptSession
# from prompt_toolkit.formatted_text import HTML
# from prompt_toolkit.shortcuts import print_formatted_text, clear

from clichatlocal.Configuration.CLIConfiguration import CLIConfiguration
from clichatlocal.Terminal import TerminalUI, PromptSessionManager
# from clichatlocal.Persistence.ConversationManager import ConversationManager
# from clichatlocal.Persistence.SystemMessagesManager import SystemMessagesManager
# from clichatlocal.Commands.CommandHandler import CommandHandler

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
        # self.system_messages_manager = SystemMessagesManager(self.config)
        # self.conversation_manager = ConversationManager(self.config)
        
        # # Initialize LLM
        # self.initialize_llm()
        
        # # Setup command handler
        # self.command_handler = CommandHandler(self)
        
        # Create prompt session
        self.prompt_session_manager = PromptSessionManager(
            self.cli_configuration)

        # Track history
        self.prompt_history = []
        self.last_prompt = None
    
    # def setup_paths(self):
    #     """Setup necessary paths for configurations"""
    #     # Get paths similar to terminal_only_infinite_loop_llama.py
    #     python_libraries_path = Path(__file__).resolve().parents[3]
    #     self.config_path = python_libraries_path.parent / "Configurations" / "HuggingFace" / "MoreTransformers"
        
    #     # Add necessary paths to sys.path
    #     corecode_directory = python_libraries_path / "CoreCode"
    #     more_transformers_directory = python_libraries_path / "HuggingFace" / "MoreTransformers"
    #     commonapi_directory = python_libraries_path / "ThirdParties" / "APIs" / "CommonAPI"
        
    #     for directory in [corecode_directory, more_transformers_directory, commonapi_directory]:
    #         if str(directory) not in sys.path:
    #             sys.path.append(str(directory))
    
    # def initialize_llm(self):
    #     """Initialize the LocalLlama3 model"""
    #     # Load configurations
    #     self.model_config = Configuration(self.config_path / "configuration.yml")
    #     self.generation_config = GenerationConfiguration(self.config_path / "generation_configuration.yml")
        
    #     # Initialize LocalLlama3
    #     self.llm = LocalLlama3(self.model_config, self.generation_config)
        
    #     # Print model info
    #     self.terminal_ui.print_info(f"Model loaded: {self.model_config.model_path}")
    #     self.terminal_ui.print_info(f"Max position embeddings: {self.llm.llm_engine.model.config.max_position_embeddings}")
    
    async def run_iterative(self):
        """Single iteration of chat interaction"""
        try:
            prompt = await self.prompt_session_manager.session.prompt_async(
                "Chat prompt (or type .help for options): "
            )
            
            if not prompt.strip():
                return True
                
            # if prompt.startswith('.'):
            #     should_continue = await self.command_handler.handle_command(prompt)
            #     return should_continue
            
            # Process regular prompt
            self.last_prompt = prompt
            self.prompt_history.append(prompt)
            
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
        
    #     # Set default system message if none is active
    #     if not self.llm.current_system_message:
    #         default_message = self.system_messages_manager.get_default_message()
    #         if default_message:
    #             self.llm.set_system_message(default_message)
    #             self.terminal_ui.print_system_message(f"Using system message: {default_message[:50]}...")
        
        async def run_async():
            continue_running = True
            while continue_running:
                continue_running = await self.run_iterative()

        asyncio.run(run_async())
    #     self.terminal_ui.print_info("Thank you for using CLIChatLocal!")
