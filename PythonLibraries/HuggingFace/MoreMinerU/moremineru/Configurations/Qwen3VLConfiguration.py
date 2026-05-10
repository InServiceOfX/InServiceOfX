from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Qwen3VLConfiguration(BaseModel):
    """Configuration for the Qwen3-VL family of vLLM-served VLMs.

    Mirrors the shape of MinerUConfiguration. Differences:
      * no MinerU-specific `backend` / `image_analysis` fields — Qwen3-VL is
        served directly through `vllm.LLM`, no MinerUClient layer.
      * adds `system_prompt` for ChatML system-role injection.
      * adds `default_sampling_params` so the wrapper can call
        `llm.chat(..., sampling_params=SamplingParams(**default_sampling_params))`
        without forcing every caller to pass full sampling configuration.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        # Pydantic v2 reserves the `model_` prefix; we use `model_path` as a
        # natural YAML key, so silence the warning.
        protected_namespaces=(),
    )

    model_path: Path = Field(
        ...,
        description=(
            "Local path to a Qwen3-VL checkpoint directory (must contain "
            "config.json + chat_template.json + safetensors shards)."
        ),
    )

    system_prompt: Optional[str] = Field(
        None,
        description=(
            "Optional system-role message prepended to every chat. Most use "
            "cases for image-grounded extraction don't need one; leave None."
        ),
    )

    vllm_engine_kwargs: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Keyword arguments forwarded to vllm.LLM(...). Common keys: "
            "dtype, max_model_len, gpu_memory_utilization, enforce_eager, "
            "tensor_parallel_size, limit_mm_per_prompt, trust_remote_code."
        ),
    )

    default_sampling_params: Dict[str, Any] = Field(
        default_factory=lambda: {"max_tokens": 1024, "temperature": 0.0},
        description=(
            "Default keyword arguments for vllm.SamplingParams when the "
            "caller doesn't pass an explicit one. Greedy decoding "
            "(temperature=0) is the right default for grounded extraction; "
            "callers doing creative tasks should override."
        ),
    )

    @field_validator("vllm_engine_kwargs")
    @classmethod
    def _reject_unknown_keys(cls, value: Dict[str, Any]) -> Dict[str, Any]:
        # Soft check — vllm.LLM accepts many kwargs; we only reject the obvious
        # MinerU-specific ones that callers might have copy-pasted.
        forbidden = {"backend", "image_analysis"}
        leaked = forbidden & set(value)
        if leaked:
            raise ValueError(
                f"vllm_engine_kwargs contains MinerU-specific keys "
                f"{sorted(leaked)}; remove them — they belong on "
                f"MinerUConfiguration, not Qwen3VLConfiguration."
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
    ) -> "Qwen3VLConfiguration":
        if not config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {config_path}"
            )

        with config_path.open("r") as file_handle:
            data = yaml.safe_load(file_handle) or {}

        required_fields: List[str] = [
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
