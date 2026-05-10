from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import Any, Dict
import yaml


class PDFExtractionConfiguration(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    input_path: Path = Field(
        ...,
        description="A .pdf file, or a directory containing .pdf files.",
    )
    output_path: Path = Field(
        ...,
        description="Directory to write extracted outputs into.",
    )
    pdf_dpi: int = Field(
        250,
        ge=72,
        le=600,
        description="DPI used when rasterizing each PDF page.",
    )
    image_format: str = Field(
        "PNG",
        description="Format for saved page images (only if save_intermediate_images).",
    )
    save_intermediate_images: bool = Field(
        True,
        description="Persist rasterized page PNGs alongside extraction outputs.",
    )
    skip_existing: bool = Field(
        True,
        description="Skip pages whose .md output already exists.",
    )

    @classmethod
    def from_yaml(cls, config_path: Path) -> "PDFExtractionConfiguration":
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
