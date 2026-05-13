from pathlib import Path
from typing import Optional
import yaml

from pydantic import BaseModel, ConfigDict, Field


class ViewerConfiguration(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, protected_namespaces=())

    mineru_output_path: Path = Field(
        ...,
        description="Directory written by CLIPDFExtraction; subdirs each hold manifest.json + page_N.md + page_N.png.",
    )
    qwen3vl_output_path: Optional[Path] = Field(
        None,
        description="Directory written by CLIPDFQwen3VLChat; same subdir structure with page_N.txt.",
    )
    colqwen_index_path: Optional[Path] = Field(
        None,
        description="Directory written by CLIPDFColQwenIndexer; used for future query support.",
    )
    host: str = Field("0.0.0.0", description="Bind address for uvicorn.")
    port: int = Field(8888, description="Bind port for uvicorn.")

    @classmethod
    def from_yaml(cls, config_path: Path) -> "ViewerConfiguration":
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        with config_path.open() as fh:
            data = yaml.safe_load(fh) or {}
        return cls(**data)
