from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, ClassVar, List, Union, Any, Dict
import yaml

@dataclass
class GenerationConfiguration:
    """Configuration class for text generation settings.

    See
    transformers/src/transformers/generation/utils.py

    In class GenerationMixin, def generate(..), this is called:
    generation_config, model_kwargs = self._prepare_generation_config(
        generation_config,
        **kwargs)

    and in def _prepare_generation_config(
    self, generation_config: optional[GenerationConfig], **kwargs)

    and in def _prepare_model_kwargs(
    model_kwarg = generation_config.update(**kwargs)

    So that in
    transformers/src/transformers/generation/configuration_utils.py

    for class GenerationConfig(..)
    def update(self, **kwargs), 

    indeed does what 
    transformers/src/transformers/generation/utils.py
    in def generate(..) claims, which is that "You can override any
    `generation_config` by passing the corresponding parameters to generate()

    Finally, see 

    transformers/src/transformers/generation/configuration_utils.py

    For possible fields to override and use here.
    """
    
    # Class constants
    DEFAULT_CONFIG_PATH: ClassVar[Path] = Path("path/to/default/generation_configuration.yml")
    FIELDS_TO_EXCLUDE: ClassVar[List[str]] = ["configuration_path", "timeout"]
    
    # Instance fields with defaults for empty construction
    configuration_path: Optional[Path] = None

    # This is needed by streamer.
    timeout: float = 60.0

    # Generation parameters
    # This default value was 8192, but as of right now, class GenerationConfig
    # in transformers/src/transformers/generation/configuration_utils.py
    # has None for max_new_tokens parameter default value.
    max_new_tokens: int = None

    do_sample: bool = False

    # Parameters that control the cache
    use_cache: bool = True

    # Parameters for manipulation of the model output logits
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 1.0
    # class Generationconfig in def __init__() uses 1.0 as default value.
    repetition_penalty: float = 1.1

    # Special tokens that can be used at generation time
    eos_token_id: List[int] = field(
        default_factory=lambda: [1280001, 128008, 128009])
    pad_token_id: int = None

    def __post_init__(self):
        """Initialize after construction."""
        # If configuration_path is provided, load from YAML
        if self.configuration_path is not None:
            self._load_from_yaml()
        
        # Validate types
        self._validate_types()
    
    @classmethod
    def from_yaml(cls, configuration_path: Optional[Path] = None) -> 'GenerationConfiguration':
        """Create a GenerationConfiguration instance from a YAML file."""
        path = configuration_path or cls.DEFAULT_CONFIG_PATH
        return cls(configuration_path=path)
    
    def _load_from_yaml(self) -> None:
        """Load configuration from YAML file."""
        path = self.configuration_path or self.DEFAULT_CONFIG_PATH
        
        try:
            with open(str(path), 'r') as f:
                data = yaml.safe_load(f)
                
            # Update instance attributes from YAML data
            for key, value in data.items():
                if hasattr(self, key) and key not in self.FIELDS_TO_EXCLUDE:
                    setattr(self, key, value)
                    
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {path}")
        except yaml.YAMLError:
            raise ValueError(f"Invalid YAML in configuration file: {path}")
    
    def _validate_types(self) -> None:
        """Validate and convert types for configuration values."""
        # Validate top_k is an integer
        if not isinstance(self.top_k, int):
            try:
                self.top_k = int(self.top_k)
            except ValueError:
                raise ValueError(f"top_k must be an integer, got {self.top_k}")
        
        # Validate top_p is a float
        if not isinstance(self.top_p, float):
            try:
                self.top_p = float(self.top_p)
            except ValueError:
                raise ValueError(f"top_p must be a float, got {self.top_p}")
        
        # Validate eos_token_id is a list
        if not isinstance(self.eos_token_id, list):
            raise ValueError(
                f"eos_token_id must be a list, got {type(self.eos_token_id)}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary, excluding specified fields."""
        return {k: v for k, v in asdict(self).items() 
                if k not in self.FIELDS_TO_EXCLUDE}
        
    def save_to_yaml(self, path: Optional[Path] = None) -> None:
        """Save configuration to YAML file."""
        save_path = path or self.configuration_path or self.DEFAULT_CONFIG_PATH
        
        with open(str(save_path), 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
