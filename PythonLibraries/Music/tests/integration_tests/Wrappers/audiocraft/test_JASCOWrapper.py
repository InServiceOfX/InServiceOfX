from corecode.Utilities import DataSubdirectories
from pathlib import Path
import torch
import torchaudio

from music.Utilities.get_stem import get_drums_stem

data_sub_dirs = DataSubdirectories()

pretrained_model_name_or_path = \
    data_sub_dirs.Models / "Generation" / "facebook" / "jasco-chords-drums-melody-1B"
if not pretrained_model_name_or_path.exists():
    pretrained_model_name_or_path = \
        data_sub_dirs.Data.parent / "Data1" / "Models" / "Generation" / \
            "facebook" / "jasco-chords-drums-melody-1B"

from audiocraft.data.audio import audio_write

from music.Configurations import MusicgenConfiguration
from music.Configurations.audiocraft import (
    JASCOConfiguration,
    JASCOGenerationConfiguration)

from music.Wrappers.audiocraft import JASCOWrapper

path_to_chords_mapping = Path(
    "/ThirdParty/audiocraft/assets/chord_to_index_mapping.pkl")

test_data_dir = Path(__file__).resolve().parents[3] / "TestData"

def test_jasco_wrapper_with_custom_melody():
    configuration = JASCOConfiguration.from_yaml(
        test_data_dir / "JASCO_configuration_minimal.yml")
    configuration.fill_defaults()
    configuration.device_map = "cuda:0"
    configuration.pretrained_model_name_or_path = \
        pretrained_model_name_or_path
    configuration.chords_mapping_path = path_to_chords_mapping

    generation_configuration = JASCOGenerationConfiguration.from_yaml(
        test_data_dir / "JASCO_generation_configuration_minimal.yml")
    generation_configuration.cfg_coef_all = 1.5
    generation_configuration.cfg_coef_txt = 2.5

    filenames = ["salience_1", "salience_2"]
    file_idx = 1
    melody_prompt_wav, melody_prompt_sample_rate = torchaudio.load(
        Path("/ThirdParty/audiocraft/assets/") / (filenames[file_idx] + ".wav"))

    chords = [
        ('N',  0.0),
        ('Eb7',  1.088000000),
        ('C#',  4.352000000),
        ('D',  4.864000000),
        ('Dm7',  6.720000000),
        ('G7',  8.256000000),
        ('Am7b5/G',  9.152000000)
    ]

    drums_wav = get_drums_stem(
        melody_prompt_wav,
        melody_prompt_sample_rate,
        configuration.device_map)

    texts = [
        '90s rock with heavy drums and hammond',
        '80s pop with groovy synth bass and drum machine',
        'folk song with leading accordion',]

    melody_path = data_sub_dirs.Data / "Private" / "Music" / \
        "05InDaClub_beginning_multif0_salience.th"
    melody_dict = torch.load(melody_path)
    #print(f"Keys in melody dictionary: {list(melody_dict.keys())}")
    
    
    # Choose the appropriate key or the first key if there's only one
    if len(melody_dict) == 1:
        # If there's only one key, just get the first value
        melody = next(iter(melody_dict.values()))
    else:
        # Otherwise, you need to specify which key to use
        # For example:
        melody = melody_dict['salience']

    print(f"Original melody shape: {melody.shape}")

    # if melody.shape[1] != 474 and melody.shape[0] != 9:
    #     if len(melody.shape) == 2:
    #         melody = torch.nn.functional.interpolate(
    #             melody.unsqueeze(0),
    #             size=(474, 9),
    #             mode='bilinear',
    #             align_corners=False
    #         ).squeeze(0).permute(1, 0)

    # print(f"Final melody shape: {melody.shape}")

    wrapper = JASCOWrapper(configuration, generation_configuration)

    output = wrapper.model.generate_music(
        descriptions=texts,
        drums_wav=drums_wav,
        drums_sample_rate=melody_prompt_sample_rate,
        chords=chords,
        melody_salience_matrix=melody,
        progress=True)

    # Save each output separately
    # Extract the i-th sample from the batch
    for i, text in enumerate(texts):
        # Create a descriptive filename based on the text prompt
        # Replace spaces with underscores and limit length
        sample = output[i]
        
        text_slug = text.replace(' ', '_')[:5]
        
        audio_write(
            f'jasco_custom_melody_drums_chords_{i}_{text_slug}',
            sample.cpu(),
            wrapper.model.sample_rate,
            strategy="loudness",
            loudness_compressor=True)