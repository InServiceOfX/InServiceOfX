from corecode.Configuration import ModelList
from morediffusers.Configurations import (
    BatchProcessingConfiguration,
    FluxGenerationConfiguration,
    NunchakuConfiguration,
    NunchakuLoRAsConfiguration,
    PipelineInputs)

class ProcessConfigurations:
    def __init__(self, application_paths):
        self._application_paths = application_paths

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
            print(f"Nunchaku configuration loaded from {path}")
        else:
            print(f"Nunchaku configuration not found at {path}")
            self.configurations["nunchaku_configuration"] = \
                NunchakuConfiguration()

        path = self._application_paths.configuration_file_paths[
            "flux_generation_configuration"]

        if path.exists():
            self.configurations["flux_generation_configuration"] = \
                FluxGenerationConfiguration.from_yaml(path)
        else:
            self.configurations["flux_generation_configuration"] = \
                FluxGenerationConfiguration()

        path = self._application_paths.configuration_file_paths[
            "nunchaku_loras_configuration"]

        if path.exists():
            self.configurations["nunchaku_loras_configuration"] = \
                NunchakuLoRAsConfiguration.from_yaml(path)
        else:
            self.configurations["nunchaku_loras_configuration"] = \
                NunchakuLoRAsConfiguration()

        path = self._application_paths.configuration_file_paths[
            "pipeline_inputs"]

        if path.exists():
            self.configurations["pipeline_inputs"] = \
                PipelineInputs.from_yaml(path)
        else:
            self.configurations["pipeline_inputs"] = \
                PipelineInputs()

        path = self._application_paths.configuration_file_paths[
            "batch_processing_configuration"]

        if path.exists():
            self.configurations["batch_processing_configuration"] = \
                BatchProcessingConfiguration.from_yaml(path)
        else:
            self.configurations["batch_processing_configuration"] = \
                BatchProcessingConfiguration()
