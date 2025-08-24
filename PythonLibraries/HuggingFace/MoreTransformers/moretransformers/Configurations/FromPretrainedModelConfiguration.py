from pydantic import BaseModel, Field, field_validator, ConfigDict
from pathlib import Path
from typing import Optional, Union, Dict, Any, Literal
import torch
import yaml

class FromPretrainedModelConfiguration(BaseModel):
    """
    Pydantic BaseModel for configuring Hugging Face's from_pretrained()
    function.

    See src/transformers/modeling_utils.py
    class PreTrainedModel(..) and def from_pretrained(..)
        
    Required field: pretrained_model_name_or_path
    Optional fields: All other parameters with sensible defaults
    """

    # Allow arbitrary types for torch.dtype
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Required field
    pretrained_model_name_or_path: Optional[Union[str, Path]] = Field(
        default=None,
        description="Path to the pretrained model (required)"
    )

    # Optional fields with defaults
    local_files_only: bool = Field(
        default=True,
        description="If True, will only try to load the tokenizer from the local cache"
    )

    force_download: Optional[bool] = Field(
        default=None,
        description=(
            "Whether to force (re-)downloading the model weights and "
            "configuration files and override cached versions if they exist.")
    )

    use_safetensors: Optional[bool] = Field(
        default=None,
        description="If set to True, will load the model using safetensors"
    )

    trust_remote_code: bool = Field(
        default=True,
        description="Whether to trust remote code when loading a model from a model repository"
    )

    # For NVIDIA GPUs for compute capability of 7.0 or greater, typically use
    # "flash_attention_2"
    # The attention implementation to use in the model (if relevant). Can be any
    # of `"eager"` (manual implementation of the attention), `"sdpa"` (using 
    # [`F.scaled_dot_product_attention`](https://pytorch.org/docs/master/generated/torch.nn.functional.scaled_dot_product_attention.html)),
    # `"flash_attention_2"` (using [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)),
    # or `"flash_attention_3"` (using [Dao-AILab/flash-attention/hopper](https://github.com/Dao-AILab/flash-attention/tree/main/hopper)).
    # By default, if available, SDPA will be used for torch>=2.1.1. The default
    # is otherwise the manual `"eager"` implementation.
    attn_implementation: Optional[Literal[
        "eager",
        "flash_attention_2",
        "flash_attention_3",
        "sdpa"]] = Field(
        default=None,
        description="Attention implementation to use. Options: 'eager', 'flash_attention_2', 'sdpa'"
    )

    torch_dtype: Optional[Union[str, torch.dtype]] = Field(
        default=None,
        description="Override the default torch.dtype and load the model under a specific dtype"
    )
    
    # For NVIDIA GPUs typically use "cuda:0"
    device_map: Optional[str] = Field(
        default=None,
        description="A map that specifies where each submodule should go"
    )
    
    # Validators
    @field_validator('pretrained_model_name_or_path')
    @classmethod
    def validate_pretrained_model_name_or_path(cls, v: Union[str, Path]) \
        -> Path:
        """Convert string to Path and validate"""
        if isinstance(v, str):
            return Path(v)
        return v
    
    @field_validator('torch_dtype', mode='before')
    @classmethod
    def validate_torch_dtype(cls, v: Any) -> Optional[torch.dtype]:
        """Convert string torch_dtype to actual torch.dtype"""
        if v is None:
            return None
        
        if isinstance(v, torch.dtype):
            return v
        
        if isinstance(v, str):
            dtype_mapping = {
                "float32": torch.float32,
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "int8": torch.int8,
                "int16": torch.int16,
                "int32": torch.int32,
                "int64": torch.int64,
                "uint8": torch.uint8,
                "bool": torch.bool,
                # Also support torch.* format
                "torch.float32": torch.float32,
                "torch.float16": torch.float16,
                "torch.bfloat16": torch.bfloat16,
                "torch.int8": torch.int8,
                "torch.int16": torch.int16,
                "torch.int32": torch.int32,
                "torch.int64": torch.int64,
                "torch.uint8": torch.uint8,
                "torch.bool": torch.bool,
            }
            
            if v in dtype_mapping:
                return dtype_mapping[v]
            else:
                raise ValueError(
                    f"Unsupported torch_dtype: {v}. "
                    f"Supported values: {list(dtype_mapping.keys())}"
                )
        
        raise ValueError(f"torch_dtype must be a string or torch.dtype, got {type(v)}")
        
    @classmethod
    def from_yaml(cls, config_path: Union[str, Path]) \
        -> 'FromPretrainedModelConfiguration':
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to the YAML configuration file
            
        Returns:
            FromPretrainedModelConfiguration instance
            
        Raises:
            FileNotFoundError: If the YAML file doesn't exist
            yaml.YAMLError: If the YAML file is malformed
            ValidationError: If the YAML data doesn't match the model schema
        """
        path = Path(config_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        try:
            with open(path, 'r') as f:
                data = yaml.safe_load(f)
            
            if data is None:
                raise ValueError(f"Configuration file {path} is empty")
            
            return cls(**data)
            
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing YAML file {path}: {str(e)}")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary, excluding None values.
        torch_dtype is preserved as either string or torch.dtype.
        
        Returns:
            Dictionary representation of the configuration
        """
        config_dict = self.model_dump()
        return {k: v for k, v in config_dict.items() if v is not None}
    
    def get_help_text(self) -> str:
        """
        Get help text showing available options and defaults.
        
        Returns:
            Formatted help text
        """
        help_text = """
FromPretrainedModelConfiguration Help:

Required Fields:
- pretrained_model_name_or_path: Path to the pretrained model (str or Path)

Optional Fields (with defaults):
- local_files_only: bool = True
- force_download: Optional[bool] = None
- use_safetensors: Optional[bool] = None
- trust_remote_code: bool = True
- attn_implementation: Optional[str] = None
  Options: "eager", "flash_attention_2", "flash_attention_3", "sdpa"
- torch_dtype: Optional[Union[str, torch.dtype]] = None
  String options: "float32", "float16", "bfloat16", "int8", "int16", "int32", "int64", "uint8", "bool"
  Also supports: "torch.float32", "torch.float16", etc.
  Or use torch.dtype directly: torch.float16, torch.bfloat16, etc.
- device_map: Optional[str] = None

Example YAML:
```yaml
pretrained_model_name_or_path: "/path/to/model"
local_files_only: true
force_download: false
use_safetensors: true
trust_remote_code: true
attn_implementation: "flash_attention_2"
torch_dtype: "float16"
device_map: "cuda:0"
```

Usage:
```python
# Load from YAML
config = FromPretrainedModelConfiguration.from_yaml("config.yml")

# Use with from_pretrained
from transformers import AutoModel
model = AutoModelForcausalLM.from_pretrained(config.model_path, **config.to_kwargs())

# Or create programmatically with torch.dtype
config = FromPretrainedModelConfiguration(
    pretrained_model_name_or_path="/path/to/model",
    torch_dtype=torch.float16,  # Direct torch.dtype
    device_map="cuda:0"
)
```
"""
        return help_text.strip()
