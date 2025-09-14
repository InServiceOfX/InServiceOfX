from corecode.Utilities import DataSubdirectories, is_model_there

from moretransformers.Configurations import FromPretrainedModelConfiguration
from transformers.models.vibevoice.vibevoice_processor import (
    VibeVoiceProcessor,
)
from transformers.models.vibevoice.modeling_vibevoice_inference import (
    VibeVoiceForConditionalGenerationInference,
)

from pathlib import Path

import pytest
import soundfile as sf
import torch

data_subdirectories = DataSubdirectories()

relative_model_path = "Models/Generative/TextToSpeech/microsoft/VibeVoice-1.5B"

is_model_downloaded, model_path = is_model_there(
    relative_model_path,
    data_subdirectories)

model_is_not_downloaded_message = f"Model {relative_model_path} not downloaded"

@pytest.mark.skipif(
        not is_model_downloaded,
        reason=model_is_not_downloaded_message)
def test_VibeVoiceForConditionalGenerationInference_inits():
    device = "cuda:0"
    dtype = torch.bfloat16 \
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported() \
            else torch.float32

    from_pretrained_model_configuration = FromPretrainedModelConfiguration(
        pretrained_model_name_or_path=model_path,
        device_map=device,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model = VibeVoiceForConditionalGenerationInference.from_pretrained(
        **from_pretrained_model_configuration.to_dict())

    assert model is not None

@pytest.mark.skipif(
        not is_model_downloaded,
        reason=model_is_not_downloaded_message)
def test_VibeVoiceProcessor_inits():
    processor = VibeVoiceProcessor.from_pretrained(
        pretrained_model_name_or_path=model_path)

    assert processor is not None

script_1 = """
Speaker 1: VibeVoice integrates seamlessly into the Transformers library.
Speaker 2: Yes, this makes it incredibly easy to use. We can just load the processor and model from the Hub.
Speaker 1: Exactly. Then we prepare the text script and provide paths to our voice samples.
Speaker 2: And finally, call the generate method. It's that simple.
"""

@pytest.mark.skipif(
        not is_model_downloaded,
        reason=model_is_not_downloaded_message)
def test_VibeVoiceProcessorProcesses():
    device = "cuda:0"

    processor = VibeVoiceProcessor.from_pretrained(
        pretrained_model_name_or_path=model_path)

    inputs = processor(
        text=[script_1,],
        return_tensors="pt",
        padding=True,
    )

    # <class 'transformers.tokenization_utils_base.BatchEncoding'>
    print(type(inputs))

    # Move inputs to correct device.
    inputs = {
        k: v.to(device) if isinstance(v, torch.Tensor) \
            else v for k, v in inputs.items()}

audio_file_path_1 = Path("/ThirdParty") / "VibeVoice" / "demo" / "voices" / \
    "en-Alice_woman.wav"
audio_file_path_2 = Path("/ThirdParty") / "VibeVoice" / "demo" / "voices" / \
    "en-Carter_man.wav"

@pytest.mark.skipif(
        not is_model_downloaded,
        reason=model_is_not_downloaded_message)
def test_VibeVoiceProcessorProcessesAudioFiles():
    device = "cuda:0"

    processor = VibeVoiceProcessor.from_pretrained(
        pretrained_model_name_or_path=model_path)

    voice_sample_paths = [str(audio_file_path_1), str(audio_file_path_2)]

    inputs = processor(
        text=[script_1,],
        voice_samples=voice_sample_paths,
        return_tensors="pt",
        padding=True,
    )

    # Move inputs to correct device.
    inputs = {
        k: v.to(device) if isinstance(v, torch.Tensor) \
            else v for k, v in inputs.items()}

    assert True

@pytest.mark.skipif(
        not is_model_downloaded,
        reason=model_is_not_downloaded_message)
def test_VibeVoiceGeneratesForText():
    """
    https://huggingface.co/microsoft/VibeVoice-1.5B/discussions/30#68ba11712f03525b46636e7e
    """
    device = "cuda:0"

    processor = VibeVoiceProcessor.from_pretrained(
        pretrained_model_name_or_path=model_path)

    voice_sample_paths = [str(audio_file_path_1), str(audio_file_path_2)]

    inputs = processor(
        text=[script_1,],
        voice_samples=voice_sample_paths,
        return_tensors="pt",
        padding=True,
    )

    inputs = {
        k: v.to(device) if isinstance(v, torch.Tensor) \
            else v for k, v in inputs.items()}

    dtype = torch.bfloat16 \
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported() \
            else torch.float32

    from_pretrained_model_configuration = FromPretrainedModelConfiguration(
        pretrained_model_name_or_path=model_path,
        device_map=device,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model = VibeVoiceForConditionalGenerationInference.from_pretrained(
        **from_pretrained_model_configuration.to_dict())

    output = model.generate(
        **inputs,
        tokenizer=processor.tokenizer,
        max_new_tokens=None,
        cfg_scale=1.3
    )

    generated_speech = output.speech_outputs[0]
    print(type(generated_speech))

    processor_sampling_rate = processor.audio_processor.sampling_rate
    print(processor_sampling_rate)
    print(type(processor_sampling_rate))
    processor.save_audio(
        generated_speech,
        "test_ForOnlyText_generated_podcast_1.5B.wav",
        sampling_rate=processor_sampling_rate,
    )

    assert True

