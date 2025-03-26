from corecode.Utilities import DataSubdirectories
from pathlib import Path

from audiocraft.data.audio import audio_write

from music.Configurations.audiocraft import JASCOConfiguration
from music.Utilities.get_stem import (
    get_drums_stem,
    get_bass_stem,
    get_vocals_stem,
    get_other_stem)
import torchaudio

data_sub_dirs = DataSubdirectories()

path_to_salience_1_wav = Path("/ThirdParty/audiocraft/assets/salience_1.wav")

test_data_dir = Path(__file__).resolve().parents[2] / "TestData"

def test_get_drums_stem():
    configuration = JASCOConfiguration.from_yaml(
        test_data_dir / "JASCO_configuration_minimal.yml")
    configuration.device_map = "cuda:0"

    wav, sample_rate = torchaudio.load(path_to_salience_1_wav)
    drums_wav = get_drums_stem(wav, sample_rate, configuration.device_map)

    audio_write(
        'salient_1_drums',
        drums_wav.cpu().squeeze(0),
        sample_rate,
        strategy="loudness",
        loudness_compressor=True)

data_private_music_path = data_sub_dirs.Data / "Private" / "Music"

if not data_private_music_path.exists():
    data_private_music_path = Path("/Data1") / "Private" / "Music"

def test_get_drum_stem_from_converted_snippet():
    configuration = JASCOConfiguration.from_yaml(
        test_data_dir / "JASCO_configuration_minimal.yml")
    configuration.device_map = "cuda:0"

    filepath = data_private_music_path / "05InDaClub_beginning.wav"
    wav, sample_rate = torchaudio.load(filepath)
    drums_wav = get_drums_stem(wav, sample_rate, configuration.device_map)

    audio_write(
        '05InDaClub_beginning_drums',
        drums_wav.cpu().squeeze(0),
        sample_rate,
        strategy="loudness",
        loudness_compressor=True)

def test_get_bass_stem_from_converted_snippet():
    configuration = JASCOConfiguration.from_yaml(
        test_data_dir / "JASCO_configuration_minimal.yml")
    configuration.device_map = "cuda:0"

    filepath = data_private_music_path / "05InDaClub_beginning.wav"
    wav, sample_rate = torchaudio.load(filepath)
    bass_wav = get_bass_stem(wav, sample_rate, configuration.device_map)

    audio_write(
        '05InDaClub_beginning_bass',
        bass_wav.cpu().squeeze(0),
        sample_rate,
        strategy="loudness",
        loudness_compressor=True)

def test_get_vocals_stem_from_converted_snippet():
    configuration = JASCOConfiguration.from_yaml(
        test_data_dir / "JASCO_configuration_minimal.yml")
    configuration.device_map = "cuda:0"

    filepath = data_private_music_path / "05InDaClub_beginning.wav"
    wav, sample_rate = torchaudio.load(filepath)
    vocals_wav = get_vocals_stem(wav, sample_rate, configuration.device_map)

    audio_write(
        '05InDaClub_beginning_vocals',
        vocals_wav.cpu().squeeze(0),
        sample_rate,
        strategy="loudness",
        loudness_compressor=True)

def test_get_other_stem_from_converted_snippet():
    configuration = JASCOConfiguration.from_yaml(
        test_data_dir / "JASCO_configuration_minimal.yml")
    configuration.device_map = "cuda:0"

    filepath = data_private_music_path / "05InDaClub_beginning.wav"
    wav, sample_rate = torchaudio.load(filepath)
    other_wav = get_other_stem(wav, sample_rate, configuration.device_map)

    audio_write(
        '05InDaClub_beginning_other',
        other_wav.cpu().squeeze(0),
        sample_rate,
        strategy="loudness",
        loudness_compressor=True)