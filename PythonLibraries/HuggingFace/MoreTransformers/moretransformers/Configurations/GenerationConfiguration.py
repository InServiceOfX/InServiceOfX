from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Union, Dict, Any
import yaml

class GenerationConfiguration(BaseModel):
    """Configuration class for text generation settings.

    Follow class GenerationMixin(ContinuousMixin) in
    src/transformers/generation/utils.py

    It says
    'Most generation-controlling parameters are set in generation_config which,
    if not passed, will be set to the mode's default generation configuration.
    You can override any generation_config by passing the corresponding
    parameters to generate(), e.g. .generate(inputs, num_beams=4,
    do_sample=True)`

    So see
    src/transformers/generation/configuration_utils.py

    In other words, see
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

    Finally, see class GenerationConfig(PushToHubMixin) in
    transformers/src/transformers/generation/configuration_utils.py
    For possible fields to override and use here.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    max_new_tokens: Optional[int] = Field(
        default=None,
        description=(
            "The maximum number of tokens to generate, ignoring number of "
            "tokens in the prompt."))

    do_sample: Optional[bool] = Field(
        default=False,
        description=(
            "Whether or not to use sampling; use greedy decoding otherwise."))

    # Parameters that control the cache
    use_cache: Optional[bool] = Field(
        default=None,
        description=(
            "Whether or not the model should use the past last key/values "
            "attentions (if applicable to the model) to speed up decoding."))

    # Parameters for manipulation of the model output logits

    temperature: Optional[float] = Field(
        default=None,
        description="The value used to module the next token probabilities.")

    top_k: Optional[int] = Field(
        default=None,
        description=(
            "The number of highest probability vocabulary tokens to keep for "
            "top-k-filtering. Defaults to 50 by huggingface's transformers."))

    top_p: Optional[float] = Field(
        default=None,
        description=(
            "If set to float < 1, only smallest set of most probable tokens"
            "with probabilities that add up to `top_p` or higher are kept for "
            "generation. Defaults to 1.0 by huggingface's transformers."))

    min_p: Optional[float] = Field(
        default=None,
        description=(
            "Minimum token probability, which will be scaled by the "
            "probability of the most likely token. It must be a value between "
            "0 and 1. Typical values are in 0.01-0.2 range, comparably "
            "selective as setting `top_p` in 0.99-0.8 range (use the opposite "
            "of normal `top_p` values"))

    repetition_penalty: Optional[float] = Field(
        default=None,
        description="1.0 means no penalty. See https://huggingface.co/papers/1909.05858)")

    # Special tokens that can be used at generation time.
    
    pad_token_id: Optional[int] = Field(
        default=None,
        description="The id of the *padding* token.")

    eos_token_id: Optional[Union[int, list[int]]] = Field(
        default=None,
        description=(
            "The id of the *end-of-sequence* token. Optionally, use a list to "
             "set multiple end-of-sequence tokens."))
    
    @classmethod
    def from_yaml(cls, config_path: Union[str, Path]) \
        -> 'GenerationConfiguration':
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to the YAML configuration file
            
        Returns:
            GenerationConfiguration instance
            
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
    
    def to_kwargs(self) -> Dict[str, Any]:
        """
        Convert configuration to kwargs for generate().
        Excludes None values.
        
        Returns:
            Dictionary of keyword arguments for generate()
        """
        config_dict = self.model_dump()
        
        # Filter out None values
        kwargs = {k: v for k, v in config_dict.items() if v is not None}
        
        return kwargs
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary, excluding None values.
        
        Returns:
            Dictionary representation of the configuration
        """
        config_dict = self.model_dump()
        return {k: v for k, v in config_dict.items() if v is not None}
    
