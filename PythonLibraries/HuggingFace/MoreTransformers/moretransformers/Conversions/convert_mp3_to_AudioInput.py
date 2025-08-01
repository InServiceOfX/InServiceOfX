from warnings import warn
import numpy as np
import torch

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    warn("soundfile is not installed")
    SOUNDFILE_AVAILABLE = False

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    warn("librosa is not installed")
    LIBROSA_AVAILABLE = False

def convert_mp3_to_AudioInput(
        mp3_file_path,
        target_sampling_rate=None,
        is_mono=True,
        is_return_torch=False):

    try:
        audio, sample_rate = sf.read(mp3_file_path)
    except Exception as err:
        warn(f"Error reading mp3 file {mp3_file_path} with soundfile: {err}")
        try:
            audio, sample_rate = librosa.load(mp3_file_path, sr=None)
        except Exception as err:
            warn(f"Error reading mp3 file {mp3_file_path} with librosa: {err}")
            return None

    if target_sampling_rate is not None:
        if sample_rate != target_sampling_rate and LIBROSA_AVAILABLE:
            audio = librosa.resample(
                audio,
                orig_sr=sample_rate,
                target_sr=target_sampling_rate
                )
            sample_rate = target_sampling_rate
    else:
        sample_rate = target_sampling_rate

    # Ensure mono audio for DiaProcessor, if needed.
    if is_mono and audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    # Normalize audio to [-1, 1] range (recommended for DAC)
    audio = audio / (np.max(np.abs(audio)) + 1e-8)

    if is_return_torch:
        audio = torch.from_numpy(audio).float()

    return audio, sample_rate