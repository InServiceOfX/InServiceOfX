from corecode.Utilities import DataSubdirectories

data_sub_dirs = DataSubdirectories()

pretrained_model_name_or_path = \
    data_sub_dirs.Models / "Generation" / "facebook" / "musicgen-medium"
if not pretrained_model_name_or_path.exists():
    pretrained_model_name_or_path = \
        data_sub_dirs.Data.parent / "Data1" / "Models" / "Generation" / \
            "facebook" / "musicgen-medium"

from transformers import pipeline
import scipy

import torch

def test_pipeline_constructs():
    """
    In __init__.py of src/transformers,
    in __init__.py of src/transformers/pipelines,
    in src/transformers/pipelines/base.py,
    class Pipeline(_ScikitCompat, PushToHubMixin)

    in __init__.py of src.transfomers/pipelines,
    def pipeline(task: str=None,
      model: Optional[..]=None,
      ...
      device: Optional[..]=None,
      device_map=None,
      torch_dtype=None) -> Pipeline

    task is only one of a few str options hardcoded in code comments.

    Do not use device and device_map at the same time as they'll conflict.

    In src/transformers/pipelines/base.py
    def __call__(self, inputs, *args, num_workers=None, ..., **kwargs)

    if args:
        logger.warning(f"Ignoring args : {args}")
    if num_works is None:
        if self._num_workers is None:
            num_workers = 0
    if batch_size is None:
        if self._batch_size is None:
            batch_size = 1

    
    """
    synthesizer = pipeline("text-to-audio", pretrained_model_name_or_path)

    # This warning is obtained:
    # `torch.nn.functional.scaled_dot_product_attention` does not support having
    # an empty attention mask. Falling back to the manual attention
    # implementation. This warning can be removed using the argument
    # `attn_implementation="eager"` when loading the model.Note that this
    # probably happens because `guidance_scale>1` or because you used 
    # `get_unconditional_inputs`. See
    # https://github.com/huggingface/transformers/issues/31189 for more
    # information.
    music = synthesizer(
        "lo-fi music with a soothing melody",
        forward_params={"do_sample": True})

    scipy.io.wavfile.write(
        "musicgen_out.wav",
        rate=music["sampling_rate"],
        data=music["audio"])

def test_pipeline_constructs_with_device():
    synthesizer = pipeline(
        "text-to-audio",
        pretrained_model_name_or_path,
        device="cuda:0",
        torch_dtype=torch.float16)

    assert synthesizer.device.type == "cuda"
    # "pt" for pytorch.
    assert synthesizer.framework == "pt"
    from transformers import Pipeline
    assert isinstance(synthesizer, Pipeline)

    music = synthesizer(
        "lo-fi music with a soothing melody",
        forward_params={"do_sample": True})

    assert isinstance(music, dict)

    scipy.io.wavfile.write(
        "musicgen_out1.wav",
        rate=music["sampling_rate"],
        data=music["audio"])

pretrained_large_model_name_or_path = \
    data_sub_dirs.Models / "Generation" / "facebook" / "musicgen-stereo-large"
if not pretrained_large_model_name_or_path.exists():
    pretrained_large_model_name_or_path = \
        data_sub_dirs.Data.parent / "Data1" / "Models" / "Generation" / \
            "facebook" / "musicgen-stereo-large"

from transformers.models.musicgen import MusicgenProcessor

def test_pipeline_and_musicgen_stereo_large():
    processor = MusicgenProcessor.from_pretrained(
        pretrained_large_model_name_or_path,
        local_files_only=True,
        device_map="cuda:0")

    inputs = processor(
        text="lo-fi music with a soothing melody",
        padding=True,
        return_tensors="pt",)

    del processor

    inputs.to("cuda:0")

    synthesizer = pipeline(
        "text-to-audio",
        pretrained_model_name_or_path,
        device="cuda:0",
        torch_dtype=torch.float16)

    # AttributeError: 'TextToAudioPipeline' object has no attribute 'text_encoder'
    #assert synthesizer.text_encoder is not None
    # AttributeError: 'TextToAudioPipeline' object has no attribute 'text_encoder_2'
    #assert synthesizer.text_encoder_2 is not None
    assert synthesizer.tokenizer is not None
    # AttributeError: 'TextToAudioPipeline' object has no attribute 'tokenizer_2'. Did you mean: 'tokenizer'
    #assert synthesizer.tokenizer_2 is not None
    assert synthesizer.transformer is not None