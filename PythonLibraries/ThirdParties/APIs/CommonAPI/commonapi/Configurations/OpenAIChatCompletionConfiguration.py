from dataclasses import dataclass, fields
from typing import Optional, List, Dict, Any, Iterable
import yaml
from pathlib import Path

@dataclass
class OpenAIChatCompletionConfiguration:
    """
    https://github.com/openai/openai-python/blob/main/src/openai/resources/chat/completions/completions.py
    """
    model: Optional[str] = None
    function_call: Optional[Any] = None
    functions: Optional[Iterable[Any]] = None
    # See
    # https://platform.openai.com/docs/guides/function-calling#function-tool-example
    # It appears to be used in function calling.
    instructions: Optional[str] = None
    max_completion_tokens: Optional[int] = None
    max_tokens: Optional[int] = None
    n: Optional[int] = None
    parallel_tool_calls: Optional[bool] = None
    # "low", "medium", or "high"; this is validated below.
    reasoning_effort: Optional[Any] = None
    response_format: Optional[Any] = None
    response_model: Optional[Any] = None
    stop: Optional[List[str]] = None
    stream: bool = False
    temperature: Optional[float] = None
    tool_choice: Optional[Any] = None
    tools: Optional[Iterable[Any]] = None
    user: Optional[str] = None

    def __post_init__(self):
        """Validate configuration after initialization"""
        if self.reasoning_effort is not None:
            valid_efforts = {"low", "medium", "high"}
            if self.reasoning_effort.lower() not in valid_efforts:
                raise ValueError(
                    f"reasoning_effort must be one of {valid_efforts} "
                    f"if provided, got {self.reasoning_effort}"
                )
            # Normalize to lowercase
            self.reasoning_effort = self.reasoning_effort.lower()
    @classmethod
    def from_yaml(cls, yaml_path: Path) -> 'OpenAIChatCompletionConfiguration':
        """Create configuration from YAML file"""
        if not yaml_path.exists():
            raise FileNotFoundError(f"YAML file not found: {yaml_path}")
            
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
            
        # Only pass fields that exist in the dataclass
        valid_fields = {f.name for f in fields(cls)}
        filtered_config = {
            k: v for k, v in config_dict.items() 
            if k in valid_fields
        }
        
        return cls(**filtered_config)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to API-compatible dictionary"""
        # Initialize with required field
        config_dict = {"model": self.model}
        
        # Special handling for tools
        if self.tools is not None:
            config_dict["tools"] = [tool for tool in self.tools]
            
        # Handle all other optional fields
        for field in fields(self):
            if field.name in ['model', 'tools']:  # Skip already handled fields
                continue
                
            value = getattr(self, field.name)
            if value is not None:
                config_dict[field.name] = value
                
        return config_dict

    

