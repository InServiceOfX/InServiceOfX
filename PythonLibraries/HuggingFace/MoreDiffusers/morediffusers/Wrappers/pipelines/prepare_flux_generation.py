from morediffusers.Wrappers import create_seed_generator
from typing import Optional

def prepare_flux_generation(
    configuration,
    generation_configuration,
    prompt: str,
    prompt_2: Optional[str] = None):

    generation_kwargs = {
        "prompt": prompt,
    }

    if prompt_2 is not None:
        generation_kwargs["prompt_2"] = prompt_2

    generation_kwargs.update(generation_configuration.get_generation_kwargs())

    generation_kwargs["generator"] = create_seed_generator(
        configuration,
        generation_configuration)

    return generation_kwargs