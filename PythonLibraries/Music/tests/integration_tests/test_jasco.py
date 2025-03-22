from corecode.Utilities import DataSubdirectories
from pathlib import Path
import torch

data_sub_dirs = DataSubdirectories()

pretrained_model_name_or_path = \
    data_sub_dirs.Models / "Generation" / "facebook" / "jasco-chords-drums-melody-1B"
if not pretrained_model_name_or_path.exists():
    pretrained_model_name_or_path = \
        data_sub_dirs.Data.parent / "Data1" / "Models" / "Generation" / \
            "facebook" / "jasco-chords-drums-melody-1B"

from audiocraft.data.audio import audio_write
from audiocraft.models import JASCO

from music.Configurations import MusicgenConfiguration

path_to_chords_mapping = Path(
    "/ThirdParty/audiocraft/assets/chord_to_index_mapping.pkl")

test_data_dir = Path(__file__).resolve().parents[1] / "TestData"

def test_jasco_example():
    """
    https://huggingface.co/facebook/jasco-chords-drums-melody-1B/blob/main/README.md
    """
    configuration = MusicgenConfiguration.from_yaml(
        test_data_dir / "musicgen_configuration_minimal.yml")

    configuration.pretrained_model_name_or_path = \
        pretrained_model_name_or_path
    configuration.fill_defaults()
    configuration.device_map = "cuda:0"
    configuration.torch_dtype = torch.float

    assert path_to_chords_mapping.exists()

    model = JASCO.get_pretrained(
        pretrained_model_name_or_path,
        chords_mapping_path=path_to_chords_mapping,
        device=configuration.device_map)

    # From jasco.py, class JASCO(BaseGenModel)
    model.set_generation_params(
        # Coefficient used in multi-source classifier free guidance - all
        # conditions term. Defaults to 5.0.
        cfg_coef_all=1.5,
        # Coefficient used in multi-source classifier free guidance - text
        # condition term. Defaults to 0.0
        cfg_coef_txt=0.5,)

    text = "Strings, woodwind, orchestral, symphony."

    chords = [
        ('C', 0.0),
        ('D', 2.0),
        ('F', 4.0),
        ('Ab', 6.0),
        ('Bb', 7.0),
        ('C', 8.0),
    ]

    output = model.generate_music(
        descriptions=[text],
        # Chord progression represented as chord, start time (sec)
        chords=chords,
        # Flag to display progress of generation process.
        progress=True)

    audio_write(
        'jasco_example.wav',
        output.cpu().squeeze(0),
        model.sample_rate,
        strategy="loudness",
        loudness_compressor=True)

import torchaudio

from audiocraft.modules.conditioners import ChromaStemConditioner

def test_jasco_with_melody():
    configuration = MusicgenConfiguration.from_yaml(
        test_data_dir / "musicgen_configuration_minimal.yml")
    configuration.pretrained_model_name_or_path = \
        pretrained_model_name_or_path
    configuration.fill_defaults()
    configuration.device_map = "cuda:0"
    configuration.torch_dtype = torch.float
    
    mp3_path = Path("/ThirdParty/audiocraft/assets/bach.mp3")
    
    # Load the melody
    melody, sample_rate = torchaudio.load(mp3_path)
    # Ensure waveform is mono (some models expect single-channel audio)
    # if melody.shape[0] > 1:
    #     melody = torch.mean(melody, dim=0, keepdim=True)

    conditioner = ChromaStemConditioner(
        output_dim=512,
        sample_rate=sample_rate,
        n_chroma=128,
        radix2_exp=12,
        duration=9,
        device=configuration.device_map)

    melody_salience_matrix = conditioner(melody)
    del conditioner

    model = JASCO.get_pretrained(
        pretrained_model_name_or_path,
        chords_mapping_path=path_to_chords_mapping,
        device=configuration.device_map)

    # From jasco.py, class JASCO(BaseGenModel)
    model.set_generation_params(
        cfg_coef_all=1.5,
        cfg_coef_txt=0.5,)
        # TypeError: FlowMatchingModel.generate() got an unexpected keyword argument 'duration'
        #duration=9)

    text = "Strings, woodwind, orchestral, symphony."
    descriptions = [text]
    
    chords = [
        ('C', 0.0),
        ('D', 2.0),
        ('F', 4.0),
        ('Ab', 6.0),
        ('Bb', 7.0),
        ('C', 8.0),
    ]
    
    # output = model.generate_music(
    #     descriptions=descriptions,
    #     chords=chords,
    #     # Pass None for melody_salience_matrix to let JASCO handle it
    #     melody_salience_matrix=None,
    #     progress=True)
    
    # audio_write(
    #     'jasco_with_melody.wav',
    #     output.cpu().squeeze(0),
    #     model.sample_rate,
    #     strategy="loudness",
    #     loudness_compressor=True)