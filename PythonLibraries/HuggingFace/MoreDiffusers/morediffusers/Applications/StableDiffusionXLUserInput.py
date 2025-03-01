from dataclasses import dataclass, field
from typing import Any, Optional
from corecode.Utilities import get_user_input

@dataclass
class StableDiffusionXLUserInput:
    generation_configuration: Any
    prompt: str = field(default_factory=lambda: get_user_input(str, "Prompt: "))
    prompt_2: Optional[str] = field(
        default_factory=lambda: get_user_input(str, "Prompt 2: ", ""))
    negative_prompt: str = field(
        default_factory=lambda: get_user_input(str, "Negative prompt: ", ""))
    negative_prompt_2: Optional[str] = field(
        default_factory=lambda: get_user_input(str, "Negative prompt 2: ", "")
    )
    num_inference_steps: Optional[int] = field(
        default_factory=lambda: get_user_input(
            int,
            "Number of inference steps, i.e. Number of steps for generation (normally 50, press Enter to skip): ",
            None
        )
    )
    base_filename: str = field(
        default_factory=lambda: get_user_input(
            str,
            "Filename base (required phrase common in filenames): "))
    iterations: int = field(
        default_factory=lambda: get_user_input(
            int,
            "Number of Iterations (i.e. number of images to generate): ",
            2
        )
    )
    guidance_scale: Optional[float] = field(
        default_factory=lambda: get_user_input(
            float,
            "Guidance Scale (press Enter to use configuration value): ",
            None
        )
    )
    guidance_scale_step: float = field(
        default_factory=lambda: get_user_input(
            float,
            "Guidance scale step value (small decimal): ",
            0.0
        )
    )

    def __post_init__(self):
        # Convert empty strings to None for optional prompts
        if self.prompt_2 == "":
            self.prompt_2 = None
        if self.negative_prompt_2 == "":
            self.negative_prompt_2 = None
            
        # Validate base_filename is not empty
        if not self.base_filename:
            raise ValueError("base_filename cannot be empty")
            
        # Update configuration if values provided
        if self.num_inference_steps is not None:
            self.generation_configuration.num_inference_steps = \
                self.num_inference_steps

        if self.guidance_scale is not None and self.guidance_scale >= 0:
            self.generation_configuration.guidance_scale = self.guidance_scale

    def get_generation_kwargs(self) -> dict:
        """Create dictionary of generation parameters for pipeline __call__"""
        kwargs = {
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "num_inference_steps": self.num_inference_steps
        }
        
        # Only add optional parameters if they are not None
        if self.prompt_2 is not None:
            kwargs["prompt_2"] = self.prompt_2
        
        if self.negative_prompt_2 is not None:
            kwargs["negative_prompt_2"] = self.negative_prompt_2
            
        if self.guidance_scale is not None:
            kwargs["guidance_scale"] = self.guidance_scale
        else:
            kwargs["guidance_scale"] = \
                self.generation_configuration.guidance_scale

        return kwargs

    def update_guidance_scale(self) -> None:
        """Update guidance scale by adding step value"""
        if self.guidance_scale is not None:
            self.guidance_scale += self.guidance_scale_step

