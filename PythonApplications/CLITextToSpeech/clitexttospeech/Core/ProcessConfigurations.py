from moretransformers.Configurations import FromPretrainedModelConfiguration
from moretransformers.Configurations.TextToSpeech import VibeVoiceConfiguration
from clitexttospeech.Configuration import CLIConfiguration

class ProcessConfigurations:
    def __init__(self, app):
        self._app = app
        self._application_paths = app._application_paths
        self._terminal_ui = app._terminal_ui
        self.configurations = {}

    def process_configurations(self):
        path = self._application_paths.configuration_file_paths[
            "vibe_voice_model_configuration"]

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