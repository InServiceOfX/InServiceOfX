from pathlib import Path
from pydantic import BaseModel, Field
from warnings import warn
import yaml

from typing import Optional

class PipelineInputs(BaseModel):

    prompt: str = Field(
        default="",
        description="The prompt to use for the image generation.")
    prompt_2: Optional[str] = Field(
        default="",
        description="The second prompt to use for the image generation.")
    negative_prompt: Optional[str] = Field(
        default="",
        description="The negative prompt to use for the image generation.")
    negative_prompt_2: Optional[str] = Field(
        default="",
        description="The second negative prompt to use for the image generation.")

    input_image_file_path: Optional[str | Path] = Field(
        default=None,
        description=(
            "The input image file path to use for the image generation."))

    @classmethod
    def from_yaml(cls, file_path: str | Path) -> "PipelineInputs":
        """
        Returns:
            PipelineInputs instance. If file doesn't exist, returns empty
            instance.
        """
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

    @classmethod
    def to_yaml(cls, file_path: str | Path) -> None:
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        data = cls.model_dump()
    
        with file_path.open("w") as f:
            yaml.dump(data, f, default_flow_style=False)

    def is_input_image_valid(self) -> bool:
        if self.input_image is None:
            return False

        if not Path(self.input_image).exists():
            return False

        return True