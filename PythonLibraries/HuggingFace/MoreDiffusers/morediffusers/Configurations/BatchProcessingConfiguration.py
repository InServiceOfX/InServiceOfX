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