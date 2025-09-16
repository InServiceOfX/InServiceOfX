from corecode.Utilities import DataSubdirectories, is_model_there
from moretransformers.Applications.TextToSpeech import VibeVoiceModelAndProcessor
from moretransformers.Configurations.TextToSpeech import VibeVoiceConfiguration
from moretransformers.Configurations import FromPretrainedModelConfiguration

from pathlib import Path

import pytest, torch

data_subdirectories = DataSubdirectories()
relative_model_path = "Models/Generative/TextToSpeech/microsoft/VibeVoice-1.5B"
is_model_downloaded, model_path = is_model_there(
    relative_model_path,
    data_subdirectories)

model_is_not_downloaded_message = f"Model {relative_model_path} not downloaded"

test_data_path = Path(__file__).parents[3] / "TestData"
test_file_path = test_data_path / "vibe_voice_configuration.yml"

@pytest.mark.skipif(
        not is_model_downloaded,
        reason=model_is_not_downloaded_message)
def test_VibeVoiceModelAndProcessor_generates():
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

    vv_configuration = VibeVoiceConfiguration.from_yaml(test_file_path)
    model_and_processor = VibeVoiceModelAndProcessor(
        from_pretrained_model_configuration,
        vv_configuration
    )

    model_and_processor.load_processor()
    #print(model_and_processor._text_files_to_strings())
    model_and_processor.process_inputs()
    model_and_processor.load_model()    
    model_and_processor.generate()
    save_path, full_hash = model_and_processor.process_and_save_output()
    assert isinstance(full_hash, str)
    assert ".wav" in str(save_path) and \
        "/Data/TestVibeVoiceOutput" in str(save_path)
