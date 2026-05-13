from pathlib import Path
from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing import Any, Dict, List
import yaml


class PDFQueryConfiguration(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    index_path: Path = Field(
        ...,
        description=(
            "Directory produced by CLIPDFColQwenIndexer, or one PDF index "
            "subdirectory containing manifest.json."
        ),
    )
    query: str | List[str] = Field(
        ...,
        description="One query string, or a list of query strings.",
    )
    top_k: int = Field(
        5,
        ge=1,
        description="Number of top page hits to return per query.",
    )
    output_path: Path = Field(
        ...,
        description="Directory or .json file path for query results.",
    )

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str | List[str]) -> str | List[str]:
        queries = [value] if isinstance(value, str) else value
        if not queries:
            raise ValueError("query must contain at least one string")
        if any(not query.strip() for query in queries):
            raise ValueError("query strings must not be empty")
        return value

    @classmethod
    def from_yaml(cls, config_path: Path) -> "PDFQueryConfiguration":
        if not config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {config_path}"
            )
        with config_path.open("r") as file_handle:
            data = yaml.safe_load(file_handle) or {}
        return cls(**data)

    def queries(self) -> list[str]:
        if isinstance(self.query, str):
            return [self.query]
        return self.query

    def resolved_output_file(self) -> Path:
        if self.output_path.suffix.lower() == ".json":
            return self.output_path
        return self.output_path / "query_results.json"

    def to_dict(self) -> Dict[str, Any]:
        data = self.model_dump()
        data["index_path"] = str(data["index_path"])
        data["output_path"] = str(data["output_path"])
        return data
