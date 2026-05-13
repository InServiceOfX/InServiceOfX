from pathlib import Path
from typing import Any, Dict

from pydantic import BaseModel, ConfigDict, Field
import yaml


class PaddleOCRVLAPIConfiguration(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    server_url: str = Field(
        "http://127.0.0.1:8080/v1",
        description="OpenAI-compatible vLLM server base URL.",
    )
    model_name: str = Field(
        "paddleocr-vl",
        description="Served model name passed to vllm serve.",
    )
    request_timeout_seconds: float = Field(
        300.0,
        gt=0,
        description="HTTP request timeout per page.",
    )
    max_tokens: int = Field(
        4096,
        ge=1,
        description="Max generated tokens per page.",
    )
    temperature: float = Field(
        0.0,
        ge=0.0,
        description="Generation temperature. Keep 0 for OCR-style extraction.",
    )

    @classmethod
    def from_yaml(cls, config_path: Path) -> "PaddleOCRVLAPIConfiguration":
        if not config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {config_path}"
            )
        with config_path.open("r") as file_handle:
            data = yaml.safe_load(file_handle) or {}
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()
