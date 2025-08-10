from corecode.FileIO import TextFile
from corecode.Utilities import (
    DataSubdirectories,
    )
from moretransformers.Configurations import GenerationConfiguration
from moretransformers.Wrappers.Models import NariLabsDia
from pathlib import Path

import pytest

data_sub_dirs = DataSubdirectories()

relative_model_path = \
    "Models/Generative/TextToSpeech/nari-labs/Dia-1.6B-0626"

is_model_downloaded = False
model_path = None

for path in data_sub_dirs.DataPaths:
    path = Path(path)
    if (path / relative_model_path).exists():
        is_model_downloaded = True
        model_path = path / relative_model_path
        break

test_data_path = Path(__file__).resolve().parents[7] / "PythonApplications" / \
    "CLITextToSpeech" / "Executables" / "dia_text.txt"

model_not_downloaded_message = \
    f"Model not downloaded: {relative_model_path} or no {test_data_path}"

is_test_data_exists = test_data_path.exists()

@pytest.mark.skipif(
        not is_model_downloaded or not is_test_data_exists,
        reason=model_not_downloaded_message)
def test_NariLabDia_works_for_text_only():
    generation_configuration = \
        NariLabsDia.create_default_generation_configuration()

    nari_labs_dia = NariLabsDia(
        model_path=model_path,
        generation_configuration=generation_configuration,
        device_map="cuda:0")

    text_0 = TextFile.load_text(test_data_path)

    inputs = nari_labs_dia.process_text_only(text_0)
    outputs = nari_labs_dia.generate_from_text_only(
        inputs,
        "test_NariLabsDia_text_only.mp3")
    assert isinstance(outputs, list)
    assert len(outputs) == 1

@pytest.mark.skipif(
        not is_model_downloaded or not is_test_data_exists,
        reason=model_not_downloaded_message)
def test_NariLabDia_works_for_voice_clone():
    """
    You will need the audio file created by running the
    test_NariLabsDia_works_for_text_only above; you need to run this test in the
    same directory as the "test_NariLabsDia_text_only.mp3" file for the test
    to work as-is.
    """
    generation_configuration = \
        NariLabsDia.create_default_generation_configuration()

    assert generation_configuration.max_new_tokens == 3072
    #generation_configuration.max_new_tokens = 6144
    generation_configuration.max_new_tokens = 12288

    nari_labs_dia = NariLabsDia(
        model_path=model_path,
        generation_configuration=generation_configuration,
        device_map="cuda:0")

    text_0 = TextFile.load_text(test_data_path)
    text_1 = (
        "[S2] Yes. I'm just saying if you use the promo code JCL. [S1] Wait a "
        "second. [S2] Promo code J 15. [S1] Okay, he broke up, which is good. "
        "[S4] Is he on drugs? Is he taking drugs? [S3] He's on drugs. [S2] No, "
        "I'm not on drugs. [S3] And he's doing a deal with [S1] This is like a "
        "PSA for not taking this stuff. You're so out of control."
    )
    input_text = text_0 + text_1

    audio_path = Path.cwd().resolve() / "test_NariLabsDia_text_only.mp3"

    inputs, prompt_len = nari_labs_dia.process_text_and_mp3(
        input_text,
        audio_path)
    outputs = nari_labs_dia.generate_from_text_and_audio(
        inputs,
        "test_NariLabsDia_voice_clone.mp3",
        prompt_len)
    assert isinstance(outputs, list)
    assert len(outputs) == 1
