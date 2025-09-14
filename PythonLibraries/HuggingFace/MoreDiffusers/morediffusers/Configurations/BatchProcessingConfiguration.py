from corecode.Utilities.Strings import format_float_for_string
from pathlib import Path
from pydantic import BaseModel, Field
from typing import Tuple
from warnings import warn
import hashlib
import json
import time
import yaml

class BatchProcessingConfiguration(BaseModel):
    number_of_images: int = Field(
        default=1,
        description="The number of images to generate.")

    base_filename: str = Field(
        default="",
        description="The base filename for the images.")

    guidance_scale_step: float = Field(
        default=0.0,
        description="The step value for the guidance scale.")

    @classmethod
    def from_yaml(cls, file_path: str | Path) -> "BatchProcessingConfiguration":
        file_path = Path(file_path)

        if not file_path.exists():
            return cls()

        try:
            with file_path.open("r") as f:
                inputs_data = yaml.safe_load(f) or {}

            return cls(**inputs_data)

        except Exception as e:
            warn(f"Warning: Could not load configuration from {file_path}: {e}")
            return cls()

    def to_yaml(self, file_path: str | Path) -> None:
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        data = self.model_dump()

        with file_path.open("w") as f:
            yaml.dump(data, f, default_flow_style=False)

    def _create_configuration_hash(
            self,
            model_name: str,
            flux_generation_configuration) -> Tuple[str, str]:
        """
        Create a unique hash based on model name, flux generation parameters,
        and current timestamp.

        Returns:
            Tuple of (full_hash, truncated_hash) where truncated_hash is
            suitable for filenames
        """
        # Get current timestamp for uniqueness
        current_timestamp = time.time()
        
        # Create a dictionary of parameters to hash
        config_dict = {
            "model_name": model_name,
            "true_cfg_scale": flux_generation_configuration.true_cfg_scale,
            "height": flux_generation_configuration.height,
            "width": flux_generation_configuration.width,
            "num_inference_steps": \
                flux_generation_configuration.num_inference_steps,
            "seed": flux_generation_configuration.seed,
            "guidance_scale": flux_generation_configuration.guidance_scale,
            "timestamp": current_timestamp,
        }
        
        # Convert to JSON string for consistent hashing
        config_json = json.dumps(config_dict, sort_keys=True, default=str)

        # Generate SHA-256 hash
        hash_object = hashlib.sha256(config_json.encode('utf-8'))
        full_hash = hash_object.hexdigest()
        
        # Truncate to 12 characters - good balance between uniqueness and
        # filename length 12 characters gives us 48 bits of entropy (12 * 4 bits
        # per hex char) This provides ~2.8 trillion unique combinations, which
        # should be sufficient
        truncated_hash = full_hash[:12]

        return full_hash, truncated_hash

    def _create_image_filename(
            self,
            index: int,
            model_name: str,
            flux_generation_configuration) -> str:

        # Generate configuration hash (both full and truncated)
        full_hash, config_hash = self._create_configuration_hash(model_name, flux_generation_configuration)

        filename = (
            f"{self.base_filename}{model_name}-Steps"
            f"{flux_generation_configuration.num_inference_steps}Iter{index}-")
        if flux_generation_configuration.guidance_scale is not None:
            filename += \
                f"Guidance{format_float_for_string(flux_generation_configuration.guidance_scale)}"
        if flux_generation_configuration.true_cfg_scale is not None:
            filename += \
                f"cfg{format_float_for_string(flux_generation_configuration.true_cfg_scale)}"

        filename += f"Iter{index}-{config_hash}"

        return filename, full_hash

    def create_and_save_image(
            self,
            index: int,
            image,
            flux_generation_configuration,
            model_name: str) -> None:

        filename, full_hash = self._create_image_filename(
            index,
            model_name,
            flux_generation_configuration)

        image_format = image.format if image.format else "PNG"

        file_path = Path(flux_generation_configuration.temporary_save_path) / \
            f"{filename}.{image_format.lower()}"

        image.save(file_path)

        return full_hash