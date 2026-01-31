from clitexttospeech.Configuration import CLIConfiguration
from morechatterbox.Configurations import ChatterboxTTSConfiguration
from morechatterbox.Configurations import TTSGenerationConfiguration \
    as ChatterboxTTSGenerationConfiguration
from moretransformers.Configurations import FromPretrainedModelConfiguration
from moretransformers.Configurations.TextToSpeech import VibeVoiceConfiguration

class ProcessConfigurations:
    def __init__(self, app):
        self._app = app
        self._application_paths = app._application_paths
        self._terminal_ui = app._terminal_ui
        self.configurations = {}

    def process_configurations(self):
        path = self._application_paths.configuration_file_paths[
            "vibe_voice_model_configuration"]

        # Vibe voice configurations

        if path.exists():
            self.configurations["vibe_voice_model_configuration"] = \
                FromPretrainedModelConfiguration.from_yaml(path)
        else:
            self._terminal_ui.print_error(
                f"Vibe voice model configuration not found at {path}")
            self.configurations["vibe_voice_model_configuration"] = \
                FromPretrainedModelConfiguration()

        path = self._application_paths.configuration_file_paths[
            "vibe_voice_configuration"]

        if path.exists():
            self.configurations["vibe_voice_configuration"] = \
                VibeVoiceConfiguration.from_yaml(path)
        else:
            self._terminal_ui.print_error(
                f"Vibe voice configuration not found at {path}")
            self.configurations["vibe_voice_configuration"] = \
                VibeVoiceConfiguration()

        # Chatterbox TTS configurations

        path = self._application_paths.configuration_file_paths[
            "chatterbox_tts_configuration"]

        if path.exists():
            self.configurations["chatterbox_tts_configuration"] = \
                ChatterboxTTSConfiguration.from_yaml(path)
        else:
            self._terminal_ui.print_error(
                f"Chatterbox TTS configuration not found at {path}")
            self.configurations["chatterbox_tts_configuration"] = \
                ChatterboxTTSConfiguration()

        path = self._application_paths.configuration_file_paths[
            "chatterbox_tts_generation_configuration"]

        if path.exists():
            self.configurations["chatterbox_tts_generation_configuration"] = \
                ChatterboxTTSGenerationConfiguration.from_yaml(path)
        else:
            self._terminal_ui.print_error(
                f"Chatterbox TTS generation configuration not found at {path}")

        path = self._application_paths.configuration_file_paths[
            "cli_configuration"]

        if path.exists():
            self.configurations["cli_configuration"] = \
                CLIConfiguration.from_yaml(path)
        else:
            self._terminal_ui.print_error(
                f"CLI configuration not found at {path}")
            self.configurations["cli_configuration"] = \
                CLIConfiguration()

    def get_vibe_voice_model_name(self):
        return self.configurations[
            "vibe_voice_model_configuration"].pretrained_model_name_or_path

    def refresh_configurations(self):
        self.process_configurations()
        self._app._vvmp.refresh_configurations(
            self.configurations["vibe_voice_model_configuration"],
            self.configurations["vibe_voice_configuration"]
        )
        self._app._chatterbox_tts_model.refresh_configurations(
            self.configurations["chatterbox_tts_configuration"],
            self.configurations["chatterbox_tts_generation_configuration"]
        )