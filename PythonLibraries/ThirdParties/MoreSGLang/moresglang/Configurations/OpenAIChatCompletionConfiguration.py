from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union
import yaml

@dataclass
class OpenAIChatCompletionConfiguration:
    reasoning_effort: Optional[str] = None
    temperature: Optional[float] = None
    max_completion_tokens: Optional[int] = None

    # While deprecated for o1 models and newer ones for OpenAI, it appears to be
    # necessary for certain models such as Qwen 2.5 and DeepSeek r1 distilled.
    # https://platform.openai.com/docs/api-reference/chat/object
    # Max number of tokens that can be generated for a completion.
    max_tokens: Optional[int] = None

    def __post_init__(self):
        """
        Validate fields after initialization. Separate this concern from parsing
        yaml.
        """
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
    def from_yaml(cls, yaml_path: Union[str, Path]) -> \
        'OpenAIChatCompletionConfiguration':
        """Load configuration from yaml file."""
        with open(yaml_path, 'r') as file:
            config_dict = yaml.safe_load(file)
            
        if not config_dict:
            raise ValueError("YAML file is empty")
                
        # Only pass keys that exist in the yaml and match our dataclass fields
        valid_keys = cls.__dataclass_fields__.keys()
        filtered_dict = {
            k: v for k, v in config_dict.items() 
            if k in valid_keys and v is not None
        }
        
        return cls(**filtered_dict)
