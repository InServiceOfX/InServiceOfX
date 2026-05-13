from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml

from pydantic import BaseModel, ConfigDict, Field


class ColQwen2_5Configuration(BaseModel):
    """Configuration for ColQwen2.5 multi-vector retrieval.

    Unlike `MinerUConfiguration` / `Qwen3VLConfiguration`, this model is
    served via ``transformers`` directly (through ``colpali-engine``), not
    via vLLM — ColQwen produces multi-vector ColBERT-style embeddings (one
    vector per image patch / text token) used with MaxSim scoring, which is
    outside vLLM's generation API.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        # Pydantic v2 reserves the `model_` prefix; `model_path` is the
        # natural YAML key — silence the warning.
        protected_namespaces=(),
    )

    model_path: Path = Field(
        ...,
        description=(
            "Local path to the ColQwen2.5 LoRA adapter directory (must "
            "contain adapter_config.json + adapter_model.safetensors). "
            "colpali-engine reads adapter_config to find the base model "
            "and stitches the LoRA on at load time."
        ),
    )

    torch_dtype: str = Field(
        "bfloat16",
        description=(
            "Forwarded to ``ColQwen2_5.from_pretrained(torch_dtype=...)``. "
            "Accepted: 'bfloat16', 'float16', 'float32', 'auto'. bfloat16 "
            "matches the LoRA training dtype and works on Ampere+ GPUs."
        ),
    )

    device_map: str = Field(
        "cuda:0",
        description=(
            "Forwarded to ``from_pretrained``. Use 'cuda:0' for single-GPU; "
            "'auto' lets HF accelerate shard across multiple devices; "
            "'mps' for Apple Silicon."
        ),
    )

    attn_implementation: Optional[str] = Field(
        None,
        description=(
            "Forwarded to ``from_pretrained``. Set 'flash_attention_2' on "
            "GPUs that have flash-attn installed. Default None lets "
            "transformers pick (SDPA on modern GPUs). The VLLMMultimodal "
            "image uninstalls flash-attn (ABI conflict with stable torch); "
            "leave this None there."
        ),
    )

    local_files_only: bool = Field(
        False,
        description=(
            "Forwarded to from_pretrained. Set true for fully offline use; "
            "the LoRA adapter must be local and the base model referenced by "
            "adapter_config.json must already exist in the HuggingFace cache."
        ),
    )

    @classmethod
    def from_yaml(
        cls,
        config_path: Path,
        validate_paths: bool = False,
    ) -> "ColQwen2_5Configuration":
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

    def validate_model_path(self) -> None:
        if not self.model_path.exists():
            raise ValueError(
                f"model_path does not exist: {self.model_path}"
            )
        adapter_config = self.model_path / "adapter_config.json"
        if not adapter_config.exists():
            raise ValueError(
                f"model_path is missing adapter_config.json "
                f"(is this a LoRA dir?): {adapter_config}"
            )

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
