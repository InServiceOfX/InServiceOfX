from corecode.Configuration import ModelList
from morediffusers.Configurations import (
    BatchProcessingConfiguration,
    FluxGenerationConfiguration,
    NunchakuConfiguration,
    NunchakuFluxControlConfiguration,
    NunchakuLoRAsConfiguration,
    PipelineInputs)

from pathlib import Path

class ProcessConfigurations:
    def __init__(self, application_paths, terminal_ui):
        self._application_paths = application_paths

        self._terminal_ui = terminal_ui

        self.configurations = {}

    def process_configurations(self):

        path = self._application_paths.configuration_file_paths["model_list"]

        if path.exists():
            self.configurations["model_list"] = \
                ModelList.from_yaml(path)
        else:
            self.configurations["model_list"] = \
                ModelList()

        path = self._application_paths.configuration_file_paths[
            "nunchaku_configuration"]

        if path.exists():
            self.configurations["nunchaku_configuration"] = \
                NunchakuConfiguration.from_yaml(path)
            self._terminal_ui.print_info(
                f"Nunchaku configuration loaded from {path}")
        else:
            self._terminal_ui.print_error(
                f"Nunchaku configuration not found at {path}")
            self.configurations["nunchaku_configuration"] = \
                NunchakuConfiguration()

        path = self._application_paths.configuration_file_paths[
            "nunchaku_flux_control_configuration"]

        if path.exists():
            self.configurations["nunchaku_flux_control_configuration"] = \
                NunchakuFluxControlConfiguration.from_yaml(path)
            self._terminal_ui.print_info(
                f"Nunchaku flux control configuration loaded from {path}")
        else:
            self._terminal_ui.print_error(
                f"Nunchaku flux control configuration not found at {path}")
            self.configurations["nunchaku_flux_control_configuration"] = None

        path = self._application_paths.configuration_file_paths[
            "flux_generation_configuration"]

        if path.exists():
            self.configurations["flux_generation_configuration"] = \
                FluxGenerationConfiguration.from_yaml(path)
            self._terminal_ui.print_info(
                f"Flux generation configuration loaded from {path}")
        else:
            self._terminal_ui.print_error(
                f"Flux generation configuration not found at {path}")
            self.configurations["flux_generation_configuration"] = \
                FluxGenerationConfiguration()

        path = self._application_paths.configuration_file_paths[
            "nunchaku_loras_configuration"]

        if path.exists():
            self.configurations["nunchaku_loras_configuration"] = \
                NunchakuLoRAsConfiguration.from_yaml(path)
            self._terminal_ui.print_info(
                f"Nunchaku LoRAs configuration loaded from {path}")
        else:
            self._terminal_ui.print_error(
                f"Nunchaku LoRAs configuration not found at {path}")
            self.configurations["nunchaku_loras_configuration"] = \
                NunchakuLoRAsConfiguration()

        path = self._application_paths.configuration_file_paths[
            "pipeline_inputs"]

        if path.exists():
            self.configurations["pipeline_inputs"] = \
                PipelineInputs.from_yaml(path)
            self._terminal_ui.print_info(
                f"Pipeline inputs configuration loaded from {path}")
        else:
            self._terminal_ui.print_error(
                f"Pipeline inputs configuration not found at {path}")
            self.configurations["pipeline_inputs"] = \
                PipelineInputs()

        path = self._application_paths.configuration_file_paths[
            "batch_processing_configuration"]

        if path.exists():
            self.configurations["batch_processing_configuration"] = \
                BatchProcessingConfiguration.from_yaml(path)
            self._terminal_ui.print_info(
                f"Batch processing configuration loaded from {path}")
        else:
            self._terminal_ui.print_error(
                f"Batch processing configuration not found at {path}")
            self.configurations["batch_processing_configuration"] = \
                BatchProcessingConfiguration()

    def get_batch_processing_configuration(self):
        return self.configurations["batch_processing_configuration"]

    def get_flux_generation_configuration(self):
        return self.configurations["flux_generation_configuration"]


    def get_model_name(self):
        return Path(
            self.configurations[
                "nunchaku_configuration"].nunchaku_model_path).name