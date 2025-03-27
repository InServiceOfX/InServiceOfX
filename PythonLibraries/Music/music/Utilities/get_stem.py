"""
https://github.com/facebookresearch/audiocraft/blob/main/demos/jasco_demo.ipynb
"""

import torch

from demucs import pretrained
from demucs.apply import apply_model
from demucs.audio import convert_audio

def _get_stem(
    wav: torch.Tensor,
    sample_rate: int,
    stem_type: str = 'drums',
    device: str = 'cuda') -> torch.Tensor:
    """Get parts of the wave that holds a specific stem, extracting it from the wav.
    Args:
        wav: Input audio tensor
        sample_rate: Sample rate of the input audio
        stem_type: Type of stem to extract ('drums', 'bass', 'vocals', or 'other')
        device: Device to run the model on
        
    Returns:
        Extracted stem as a torch.Tensor
        
    https://github.com/facebookresearch/audiocraft/blob/main/demos/jasco_demo.ipynb
    """
    demucs_model = pretrained.get_model('htdemucs').to(device)
    wav = convert_audio(
        wav,
        sample_rate,
        demucs_model.samplerate,
        demucs_model.audio_channels)
    stems = apply_model(
        demucs_model,
        wav.cuda().unsqueeze(0),
        device=device).squeeze(0)
    # Extract relevant stem
    stem = stems[demucs_model.sources.index(stem_type)]
    return convert_audio(
        stem.cpu(),
        demucs_model.samplerate,
        sample_rate,
        1)

def get_drums_stem(
    wav: torch.Tensor,
    sample_rate: int,
    device: str = 'cuda') -> torch.Tensor:
    """Get parts of the wave that holds the drums, extracting the main stems
    from the wav."""
    return _get_stem(wav, sample_rate, 'drums', device)

def get_bass_stem(
    wav: torch.Tensor,
    sample_rate: int,
    device: str = 'cuda') -> torch.Tensor:
    """Get parts of the wave that holds the bass, extracting the main stems from
    the wav."""
    return _get_stem(wav, sample_rate, 'bass', device)

def get_vocals_stem(
    wav: torch.Tensor,
    sample_rate: int,
    device: str = 'cuda') -> torch.Tensor:
    """Get parts of the wave that holds the vocals, extracting the main stems
    from the wav."""
    return _get_stem(wav, sample_rate, 'vocals', device)

def get_other_stem(
    wav: torch.Tensor,
    sample_rate: int,
    device: str = 'cuda') -> torch.Tensor:
    """Get parts of the wave that holds other instruments, extracting the main
    stems from the wav."""
    return _get_stem(wav, sample_rate, 'other', device)
