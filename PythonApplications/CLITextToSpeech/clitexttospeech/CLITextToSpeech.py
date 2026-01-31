from clitexttospeech.Core import (
    GenerateWithChatterbox,
    GenerateWithVibeVoice,
    ProcessConfigurations)
from clitexttospeech.Terminal import CommandHandler
from clitexttospeech.Terminal import PromptSessionsManager
from clitexttospeech.Terminal import TerminalUI

from morechatterbox.Wrappers import ChatterboxTTSModel
from moretransformers.Applications.TextToSpeech \
    import VibeVoiceModelAndProcessor

from pathlib import Path

class CLITextToSpeech:
    def __init__(self, application_paths):
        self._application_paths = application_paths
        self._terminal_ui = TerminalUI()

        self._process_configurations = ProcessConfigurations(self)
        self._process_configurations.process_configurations()

        self._generate_with_chatterbox = GenerateWithChatterbox(self)
        self._generate_with_vibe_voice = GenerateWithVibeVoice(self)

        self._command_handler = CommandHandler(self)
        
        self._psm = PromptSessionsManager(self)

        self._vvmp = VibeVoiceModelAndProcessor(
            self._process_configurations.configurations[
                "vibe_voice_model_configuration"],
            self._process_configurations.configurations[
                "vibe_voice_configuration"])

        self._chatterbox_tts_model = ChatterboxTTSModel(
            self._process_configurations.configurations[
                "chatterbox_tts_configuration"],
            self._process_configurations.configurations[
                "chatterbox_tts_generation_configuration"])

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
                    self._terminal_ui.print_processing(prompt)

                return continue_running
            
            # Treat as text input for generation
            self._terminal_ui.print_processing(prompt)
            return True
            
        except KeyboardInterrupt:
            self._terminal_ui.print_goodbye()
            return False
        except Exception as e:
            self._terminal_ui.print_error(str(e))
            return True
    
    def run(self):
        """Main run loop"""
        self._terminal_ui.print_header("CLITextToSpeech - Text-to-Speech Tool")
        continue_running = True
        while continue_running:
            continue_running = self.run_iterative()