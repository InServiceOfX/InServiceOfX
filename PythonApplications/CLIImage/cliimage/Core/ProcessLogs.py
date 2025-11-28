from morediffusers.Utilities.Logging import (
    NunchakuGenerationLogger)

from pathlib import Path

class ProcessLogs:
    def __init__(self, application_paths):
        self.nunchaku_generation_logger = NunchakuGenerationLogger(
            application_paths.logs_file_paths["nunchaku_generation_logs"])

    def log_nunchaku_generation(
        self,
        nunchaku_config,
        flux_gen_config,
        pipeline_inputs,
        loras_config,
        nunchaku_model_index,
        generation_hash,
        truncated_generation_hash,
    ) -> None:
        """
        This is a dummy passthrough for the actual logging function for single
        prompt generation with nunchaku.
        """
        self.nunchaku_generation_logger.log_generation(
            nunchaku_config,
            flux_gen_config,
            pipeline_inputs,
            loras_config,
            nunchaku_model_index,
            generation_hash,
            truncated_generation_hash)
