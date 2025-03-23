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
from music.Configurations.audiocraft import (
    JASCOConfiguration,
    JASCOGenerationConfiguration)

from music.Wrappers.audiocraft import JASCOWrapper

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

    #text = "Strings, woodwind, orchestral, symphony."
    text = "Lo fi hip hop with 2000s boom bap throwback style with Scott Storch chords"

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

import librosa
import numpy as np

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

    # Extract fundamental frequency (F0) from the audio, use pre-trained model
    # such as pyin.
    f0, voiced_flag, voiced_probs = librosa.core.piptrack(
        melody,
        fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C7'),
        sr=sample_rate)

    # Replace NaNs with 0s
    f0 = np.nan_to_num(f0)

    # 88 for piano keys.
    melody_bins = 88

    # Initialize salience matrix.
    melody_salience_matrix = np.zeros((melody_bins, len(f0)))

    for t, frequency in enumerate(f0):
        # If there's a detected frequency,
        if frequency > 0:
            midi_note = librosa.hz_to_midi(frequency)
            # Assuming 88 keys starting from A0 (MIDI 21)
            bin_index = int(midi_note) - 21
            melody_salience_matrix[bin_index, t] = 1    

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

    text = "Lo fi hip hop 2000s throwback style"
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

# https://github.com/facebookresearch/audiocraft/blob/main/demos/jasco_demo.ipynb
def test_jasco_wrapper_text_conditional_generation():
    configuration = JASCOConfiguration.from_yaml(
        test_data_dir / "JASCO_configuration_minimal.yml")
    configuration.fill_defaults()
    configuration.device_map = "cuda:0"
    configuration.torch_dtype = torch.float
    configuration.pretrained_model_name_or_path = \
        pretrained_model_name_or_path
    configuration.chords_mapping_path = path_to_chords_mapping

    generation_configuration = JASCOGenerationConfiguration.from_yaml(
        test_data_dir / "JASCO_generation_configuration_minimal.yml")
    generation_configuration.cfg_coef_all = 0.0
    generation_configuration.cfg_coef_txt = 5.0

    wrapper = JASCOWrapper(configuration, generation_configuration)

    text = "Funky groove with electric piano playing blue chords rhythmically"
    descriptions = [text]

    output = wrapper.model.generate(descriptions=descriptions, progress=True)

    audio_write(
        'jasco_text_conditional_generation',
        output.cpu().squeeze(0),
        wrapper.model.sample_rate,
        strategy="loudness",
        loudness_compressor=True)

from audiocraft.utils.notebook import display_audio

# https://github.com/facebookresearch/audiocraft/blob/main/demos/jasco_demo.ipynb
def test_jasco_wrapper_chord_conditional_generation():
    configuration = JASCOConfiguration.from_yaml(
        test_data_dir / "JASCO_configuration_minimal.yml")
    configuration.fill_defaults()
    configuration.device_map = "cuda:0"
    configuration.torch_dtype = torch.float
    configuration.pretrained_model_name_or_path = \
        pretrained_model_name_or_path
    configuration.chords_mapping_path = path_to_chords_mapping

    generation_configuration = JASCOGenerationConfiguration.from_yaml(
        test_data_dir / "JASCO_generation_configuration_minimal.yml")
    generation_configuration.cfg_coef_all = 1.5
    generation_configuration.cfg_coef_txt = 3.0

    wrapper = JASCOWrapper(configuration, generation_configuration)

    # set textual prompt
    text = "Strings, woodwind, orchestral, symphony."
    descriptions = [text]

    # define chord progression
    chords = [('C', 0.0), ('D', 2.0), ('F', 4.0), ('Ab', 6.0), ('Bb', 7.0), ('C', 8.0)]

    output = wrapper.model.generate_music(
        descriptions=descriptions,
        chords=chords,
        progress=True)

    # This results in
    # <IPython.lib.display.Audio object>   2 /    300
    # display_audio(
    #     output.cpu(),
    #     sample_rate=wrapper.model.compression_model.sample_rate)

    audio_write(
        'jasco_chord_conditional_generation',
        output.cpu().squeeze(0),
        wrapper.model.sample_rate,
        strategy="loudness",
        loudness_compressor=True)

# https://github.com/facebookresearch/audiocraft/blob/main/demos/jasco_demo.ipynb
def test_jasco_wrapper_drums_conditional_generation():
    configuration = JASCOConfiguration.from_yaml(
        test_data_dir / "JASCO_configuration_minimal.yml")
    configuration.fill_defaults()
    configuration.device_map = "cuda:0"
    configuration.torch_dtype = torch.float
    configuration.pretrained_model_name_or_path = \
        pretrained_model_name_or_path
    configuration.chords_mapping_path = path_to_chords_mapping

    generation_configuration = JASCOGenerationConfiguration.from_yaml(
        test_data_dir / "JASCO_generation_configuration_minimal.yml")
    generation_configuration.cfg_coef_all = 1.5
    generation_configuration.cfg_coef_txt = 3.0

    wrapper = JASCOWrapper(configuration, generation_configuration)

    # set textual prompt
    text = "distortion guitars, heavy rock, catchy beat"
    descriptions = [text]

    drums_waveform, sample_rate = torchaudio.load(
        Path("/ThirdParty/audiocraft/assets/sep_drums_1.mp3"))

    output = wrapper.model.generate_music(
        descriptions=descriptions,
        drums_wav=drums_waveform,
        drums_sample_rate=sample_rate,
        progress=True)

    audio_write(
        'jasco_drums_conditional_generation',
        output.cpu().squeeze(0),
        wrapper.model.sample_rate,
        strategy="loudness",
        loudness_compressor=True)