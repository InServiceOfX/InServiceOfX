from corecode.Utilities.Strings import format_float_for_string
from pathlib import Path
from pydantic import BaseModel, Field
from warnings import warn
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

    def _create_image_filename(
            self,
            index: int,
            model_name: str,
            flux_generation_configuration) -> str:

        filename = ""

        if flux_generation_configuration.guidance_scale is None:
            filename = (
                f"{self.base_filename}{model_name}-"
                f"Steps{flux_generation_configuration.num_inference_steps}Iter{index}"
            )

        else:
            filename = (
                f"{self.base_filename}{model_name}-"
                f"Steps{flux_generation_configuration.num_inference_steps}Iter{index}Guidance{format_float_for_string(flux_generation_configuration.guidance_scale)}"
            )

        return filename

    def create_and_save_image(
            self,
            index: int,
            image,
            flux_generation_configuration,
            model_name: str) -> None:

        filename = self._create_image_filename(
            index,
            model_name,
            flux_generation_configuration)

        image_format = image.format if image.format else "PNG"

        file_path = Path(flux_generation_configuration.temporary_save_path) / \
            f"{filename}.{image_format.lower()}"

        image.save(file_path)