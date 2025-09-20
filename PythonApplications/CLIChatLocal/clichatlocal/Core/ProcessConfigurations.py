from moretransformers.Configurations import (
    FromPretrainedModelConfiguration,
    FromPretrainedTokenizerConfiguration,
    GenerationConfiguration)

from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.shortcuts import print_formatted_text
from prompt_toolkit.styles import Style

class ProcessConfigurations:
    def __init__(self, application_paths, terminal_ui):
        self._application_paths = application_paths
        self._terminal_ui = terminal_ui

        self.configurations = {}

        self._setup_styles()

    # Copy the following style from TerminalUI.py; we need to have this class
    # initialize first before we can use TerminalUI.
    def _setup_styles(self):
        self.style = Style.from_dict({
            "info": "fg:#3498db",
            "error": "fg:#e74c3c bold",
        })

    # Copy the following function implementations from TerminalUI.py; we need to
    # have this class initialize first before we can use TerminalUI.

    def _print_info(self, message: str):
        print_formatted_text(HTML(f"<info>ℹ {message}</info>"))

    def _print_error(self, message: str):
        print_formatted_text(HTML(f"<error>✗ {message}</error>"))

    def process_configurations(self):
        from_pretrained_model_path = \
            self._application_paths.configuration_file_paths[
                "from_pretrained_model"]

        if from_pretrained_model_path.exists():
            from_pretrained_model_configuration = \
                FromPretrainedModelConfiguration.from_yaml(
                    from_pretrained_model_path)
            self._print_info(
                f"From pretrained model configuration loaded from {from_pretrained_model_path}")
        else:
            from_pretrained_model_configuration = \
                FromPretrainedModelConfiguration()
            self._print_error(
                f"From pretrained model configuration not found at {from_pretrained_model_path};"
                " using default configuration")

        from_pretrained_tokenizer_path = \
            self._application_paths.configuration_file_paths[
                "from_pretrained_tokenizer"]

        if from_pretrained_tokenizer_path.exists():
            from_pretrained_tokenizer_configuration = \
                FromPretrainedTokenizerConfiguration.from_yaml(
                    from_pretrained_tokenizer_path)
            self._print_info(
                f"From pretrained tokenizer configuration loaded from {from_pretrained_tokenizer_path}")
        elif from_pretrained_model_configuration.pretrained_model_name_or_path is not None:
            from_pretrained_tokenizer_configuration = \
                FromPretrainedTokenizerConfiguration(
                    pretrained_model_name_or_path=\
                        from_pretrained_model_configuration.pretrained_model_name_or_path)
            self._print_info(
                f"From pretrained tokenizer configuration loaded from {from_pretrained_model_configuration.pretrained_model_name_or_path}")
        else:
            from_pretrained_tokenizer_configuration = \
                FromPretrainedTokenizerConfiguration()
            self._print_info(
                (
                    f"From pretrained tokenizer configuration not found at {from_pretrained_tokenizer_path};",
                    " using default configuration")
            )

        generation_configuration_path = \
            self._application_paths.configuration_file_paths["generation"]

        if generation_configuration_path.exists():
            generation_configuration = \
                GenerationConfiguration.from_yaml(generation_configuration_path)
            self._print_info(
                f"Generation configuration loaded from {generation_configuration_path}")
        else:
            generation_configuration = GenerationConfiguration()
            self._print_error(
                "Generation configuration not found at {generation_configuration_path};"
                " using default configuration")

        self.configurations["from_pretrained_model_configuration"] = \
            from_pretrained_model_configuration
        self.configurations["from_pretrained_tokenizer_configuration"] = \
            from_pretrained_tokenizer_configuration
        self.configurations["generation_configuration"] = \
            generation_configuration

        return (
            from_pretrained_model_configuration,
            from_pretrained_tokenizer_configuration,
            generation_configuration)