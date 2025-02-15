from pathlib import Path
import yaml
from pydantic import BaseModel, Field
from typing import Union, Optional, Dict

class GenerationConfiguration(BaseModel):
    temporary_save_path: str = Field(default="/Data/Private")
    input_images_directory: Optional[str] = Field(default=None)
    # From
    # https://docs.lumalabs.ai/docs/python-video-generation#ray-2
    # https://docs.lumalabs.ai/docs/python-video-generation#usage-example
    # From lumaai-python, resources/generations/generations.py
    # aspect_ratio: Literal["1:1", "16:9", "9:16", "4:3", "3:4", "21:9", "9:21"] | NotGiven = NOT_GIVEN,
    aspect_ratio: Optional[str] = Field(default=None)
    loop: Optional[bool] = Field(default=None)
    # model: Literal["ray-1-6", "ray-2"] | NotGiven = NOT_GIVEN,
    model: Optional[str] = Field(default=None)
    # resolution: Union[Literal["540p", "720p"], str] | NotGiven = NOT_GIVEN,
    resolution: Optional[str] = Field(default=None)
    # duration: Union[Literal["5s", "9s"], str] | NotGiven = NOT_GIVEN,
    duration: Optional[str] = Field(default=None)
    loop: Optional[bool] = Field(default=None)

    @classmethod
    def from_yaml(cls, configuration_path: Union[Path, str]) -> \
        'GenerationConfiguration':
        with open(str(configuration_path), 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
    def to_api_kwargs(self) -> dict:
        """
        Convert configuration to API kwargs, excluding None values and
        configuration paths
        """
        api_kwargs = self.model_dump(
            exclude={'temporary_save_path', 'input_images_directory'})
        return {k: v for k, v in api_kwargs.items() if v is not None}
