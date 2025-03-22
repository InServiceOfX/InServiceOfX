from corecode.Utilities import clear_torch_cache_and_collect_garbage
from transformers.models.musicgen import MusicgenProcessor

from typing import Union, List

def from_prompt_and_processor(
    prompt: Union[str, List[str]],
    configuration):

    processor = MusicgenProcessor.from_pretrained(
        configuration.pretrained_model_name_or_path,
        local_files_only=True,
        device_map=configuration.device_map)

    inputs = processor(
        text=prompt,
        padding=True,
        return_tensors="pt")

    del processor
    clear_torch_cache_and_collect_garbage()

    return inputs