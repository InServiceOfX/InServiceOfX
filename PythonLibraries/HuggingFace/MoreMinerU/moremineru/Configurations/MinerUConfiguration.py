from pathlib import Path
from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import Any, Dict, Optional
import yaml


class MinerUConfiguration(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        # Pydantic v2 treats `model_` as a protected prefix; we use
        # `model_path` as a natural YAML key so silence that warning.
        protected_namespaces=(),
    )

    model_path: Path = Field(
        ...,
        description=(
            "Local path to the MinerU2.5-Pro model weights "
            "(directory with config.json + model.safetensors)."
        ),
    )

    backend: str = Field(
        "vllm-engine",
        description=(
            "MinerUClient backend. Currently only 'vllm-engine' is wired up "
            "in this wrapper; transformers/lmdeploy/mlx would require their "
            "own loader paths."
        ),
    )

    image_analysis: bool = Field(
        False,
        description=(
            "Pass-through to MinerUClient.image_analysis. When True the model "
            "additionally analyses figures/charts; default False matches the "
            "model card's recommended document-extraction setup."
        ),
    )

    vllm_engine_kwargs: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Keyword arguments forwarded directly to vllm.LLM(...). "
            "Common keys: dtype ('bfloat16'/'float16'/'auto'), max_model_len, "
            "gpu_memory_utilization, enforce_eager, tensor_parallel_size, "
            "trust_remote_code."
        ),
    )

    @field_validator("backend")
    @classmethod
    def validate_backend(cls, value: str) -> str:
        allowed = {"vllm-engine"}
        if value not in allowed:
            raise ValueError(
                f"backend must be one of {sorted(allowed)}, got {value!r}"
            )
        return value

    def validate_model_path(self) -> None:
        if not self.model_path.exists():
            raise ValueError(
                f"model_path does not exist: {self.model_path}"
            )
        config_json = self.model_path / "config.json"
        if not config_json.exists():
            raise ValueError(
                f"model_path is missing config.json: {config_json}"
            )

    @classmethod
    def from_yaml(
        cls,
        config_path: Path,
        validate_paths: bool = False,
    ) -> "MinerUConfiguration":
        if not config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {config_path}"
            )

        with config_path.open("r") as file_handle:
            data = yaml.safe_load(file_handle) or {}

        required_fields = [
            field_name
            for field_name, field_info in cls.model_fields.items()
            if field_info.is_required()
        ]
        missing_fields = [
            name for name in required_fields if name not in data
        ]
        if missing_fields:
            raise ValueError(
                f"Missing required fields in configuration: {missing_fields}"
            )

        config = cls(**data)

        if validate_paths:
            config.validate_model_path()

        return config

    def to_dict(self) -> Dict[str, Any]:
        data = self.model_dump()
        data["model_path"] = str(data["model_path"])
        return data

    def save_yaml(self, config_path: Path) -> None:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with config_path.open("w") as file_handle:
            yaml.dump(
                self.to_dict(),
                file_handle,
                default_flow_style=False,
                indent=2,
            )

    def get_tensor_parallel_size(self) -> int:
        return int(self.vllm_engine_kwargs.get("tensor_parallel_size", 1))

    def get_dtype(self) -> Optional[str]:
        return self.vllm_engine_kwargs.get("dtype")
