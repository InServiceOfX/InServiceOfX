from moretransformers.Configurations.TextToSpeech import VibeVoiceConfiguration

from pathlib import Path

test_data_path = Path(__file__).parents[3] / "TestData"
test_file_path = test_data_path / "vibe_voice_configuration.yml"

def test_VibeVoiceConfiguration_from_yaml():
    configuration = VibeVoiceConfiguration.from_yaml(test_file_path)
    assert configuration is not None
    assert configuration.audio_file_paths == [
        Path("/ThirdParty/VibeVoice/demo/voices/en-Frank_man.wav"),
        Path("/ThirdParty/VibeVoice/demo/voices/en-Mary_woman_bgm.wav")
    ]
    assert configuration.text_file_paths == [
        Path("/InServiceOfX/PythonLibraries/HuggingFace/MoreTransformers/tests/TestData/VibeVoiceExampleDialogueAP.txt")
    ]
    assert configuration.cfg_scale == 1.3
    assert configuration.max_new_tokens is None
    assert configuration.directory_path_to_save == "/Data/"
    assert configuration.base_saved_filename == "TestVibeVoiceOutput"