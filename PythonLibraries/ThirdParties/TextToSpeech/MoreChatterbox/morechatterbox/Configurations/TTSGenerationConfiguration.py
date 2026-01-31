from pathlib import Path
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional, Tuple, Union

import json
import time
import yaml

class TTSGenerationConfiguration(BaseModel):
    """
    Based upon the following function signature of in class ChatterboxTTS:
    def generate(
        self,
        text,
        repetition_penalty=1.2,
        min_p=0.05,
        top_p=1.0,
        audio_prompt_path=None,
        exaggeration=0.5,
        cfg_weight=0.5,
        temperature=0.8,
    ):
    in
    https://github.com/resemble-ai/chatterbox/blob/master/src/chatterbox/tts.py#L208
    """
    repetition_penalty: Optional[float] = Field(
        default=1.2,
        description=(
            "Reduces probabilty of recently generated tokens. >1.0 penalizes "
            "repetition;<1.0 encourages it."),
    )

    min_p: Optional[float] = Field(
        default=0.05,
        description=(
            "Filters tokens with probabilty below min_p * max_probability."
            "Removes low-probability tokens. Higher (0.1-0.2) = more focused; "
            "0.0 disables. Works well with higher temperature."),
    )

    top_p: Optional[float] = Field(
        default=1.0,
        description=(
            "Nucleus sampling -- keeps tokens whose cumulative probability "
            "reaches to p_p. Lower (0.8-0.95)=more focused; 1.0 disables. "
            "Often used with top_k."),
    )

    exaggeration: Optional[float] = Field(
        default=0.5,
        description=(
            "Controls emotional expressiveness. 0.5 is neutral; higher=more "
            "expressiveness. lower=more neutral."),
    )

    cfg_weight: Optional[float] = Field(
        default=0.5,
        description=("Classifier-free guidance weight."),
    )

    temperature: Optional[float] = Field(
        default=0.8,
        description=("Controls randomness."),
    )

    max_new_tokens: Optional[int] = Field(
        default=1000,
        description=("Maximum number of new speech tokens to generate."),
    )

    @classmethod
    def from_yaml(cls, file_path: Union[str, Path]) -> "TTSGenerationConfiguration":
        with open(file_path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary, excluding None values."""
        config_dict = self.model_dump()
        return {k: v for k, v in config_dict.items() if v is not None}