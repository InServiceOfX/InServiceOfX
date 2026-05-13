from pathlib import Path
from pydantic import BaseModel, ConfigDict, Field
from typing import Any, Dict
import yaml


class PDFChatConfiguration(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    input_path: Path = Field(...)
    output_path: Path = Field(...)
    prompt: str = Field(...)
    pdf_dpi: int = Field(250, ge=72, le=600)
    image_format: str = Field("PNG")
    save_intermediate_images: bool = Field(True)
    skip_existing: bool = Field(True)

    @classmethod
    def from_yaml(cls, config_path: Path) -> "PDFChatConfiguration":
        if not config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {config_path}"
            )
        with config_path.open("r") as file_handle:
            data = yaml.safe_load(file_handle) or {}
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        data = self.model_dump()
        data["input_path"] = str(data["input_path"])
        data["output_path"] = str(data["output_path"])
        return data

    def list_input_pdfs(self) -> list[Path]:
        if self.input_path.is_file():
            if self.input_path.suffix.lower() != ".pdf":
                raise ValueError(
                    f"input_path is a file but not a .pdf: {self.input_path}"
                )
            return [self.input_path]
        if self.input_path.is_dir():
            return sorted(self.input_path.glob("*.pdf"))
        raise FileNotFoundError(
            f"input_path does not exist: {self.input_path}"
        )
