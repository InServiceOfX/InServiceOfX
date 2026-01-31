from chatterbox.tts import ChatterboxTTS
from corecode.FileIO import TextFile
from morechatterbox.Configurations import (
    ChatterboxTTSConfiguration,
    TTSGenerationConfiguration
)
from pathlib import Path
from typing import List, Optional, Tuple
from warnings import warn

import torch, torchaudio

class ChatterboxTTSModel:
    """Wrapper for Chatterbox TTS model with configuration-based generation."""
    
    def __init__(
        self,
        configuration: ChatterboxTTSConfiguration,
        tts_generation_configuration: Optional[
            TTSGenerationConfiguration] = None,
    ):
        self.configuration = configuration
        if tts_generation_configuration is None:
            self.tts_generation_configuration = TTSGenerationConfiguration()
        else:
            self.tts_generation_configuration = tts_generation_configuration

        if self.configuration.model_dir is None:
            raise ValueError(
                "ChatterboxTTSModel.__init__: "
                "model_dir is required in configuration"
            )
        
        self._model: Optional[ChatterboxTTS] = None
        self._sample_rate: Optional[int] = None

    def refresh_configurations(
        self,
        configuration: ChatterboxTTSConfiguration,
        tts_generation_configuration: Optional[
            TTSGenerationConfiguration] = None,
        ):
        self.configuration = configuration
        if tts_generation_configuration is not None:
            self.tts_generation_configuration = tts_generation_configuration

    def is_model_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None

    def load_model(self) -> None:
        """Load the Chatterbox TTS model."""
        if self._model is not None:
            return
        
        self._model = ChatterboxTTS.from_local(
            ckpt_dir=str(self.configuration.model_dir),
            device=self.configuration.device
        )
        
        # Get sample rate from model
        self._sample_rate = self._model.sr

    def unload_model(self) -> None:
        """Unload the model to free memory."""
        if self._model is not None:
            del self._model
            self._model = None
            self._sample_rate = None

    def _text_file_to_string(self) -> str:
        """Read text from text file path."""
        return TextFile.load_text(self.configuration.text_file_path)

    def _text_file_paths_to_strings(self) -> List[str]:
        """Read text from text file paths."""
        return [TextFile.load_text(path) \
            for path in self.configuration.text_file_paths]

    def generate(self, text: Optional[str] = None) -> Tuple[torch.Tensor, int]:
        """
        Generate speech from text using voice cloning.
        """
        if not self.is_model_loaded():
            self.load_model()

        text_to_use = text if text is not None else self._text_file_to_string()
        if not text_to_use:
            raise ValueError("No text found in text file")

        audio_prompt_path = None
        if self.configuration.audio_prompt_path.exists():
            audio_prompt_path = self.configuration.audio_prompt_path

        wav = None
        generation_configuration = self.tts_generation_configuration.to_dict()
        if audio_prompt_path is None:
            wav = self._model.generate(text_to_use, **generation_configuration)
        else:
            wav = self._model.generate(
                text_to_use,
                audio_prompt_path=str(audio_prompt_path),
                **generation_configuration
            )

        # Use configured sample rate or model's default
        sample_rate = (
            self._model.sr
            or self.configuration.sample_rate 
            or self._sample_rate 
        )

        return wav, sample_rate

    def save_output(
        self,
        wav: torch.Tensor,
        sample_rate: int,
        filename: Optional[str] = None,
    ) -> Path:
        """
        Save generated audio to file.
        
        Args:
            wav: Audio tensor
            sample_rate: Sample rate
            filename: Optional filename (uses config default if None)
        
        Returns:
            Path to saved file
        """
        if self.configuration.directory_path_to_save is None:
            raise ValueError(
                "directory_path_to_save must be set in configuration"
            )

        output_dir = Path(self.configuration.directory_path_to_save)
        output_dir.mkdir(parents=True, exist_ok=True)

        if filename is None:
            filename, _ = self.configuration.create_save_filename()
        
        output_path = output_dir / filename

        torchaudio.save(
            str(output_path),
            wav,
            sample_rate
        )
        
        return output_path

    def generate_and_save(self) -> Path:
        """
        Generate speech and save to files.
        
        Returns:
            List of paths to saved audio files
        """
        wav, sample_rate = self.generate()

        return self.save_output(wav, sample_rate)

    def generate_and_save_for_text_file_paths(self) -> List[Path]:
        """
        Generate speech and save to files for text file paths.
        
        Returns:
            List of paths to saved audio files
        """
        if self.configuration.text_file_paths is None:
            warn("text_file_paths is not set in configuration")
            return []

        texts = self._text_file_paths_to_strings()
        if not texts:
            warn("No text found in text file paths")
            return []

        filenames = \
            self.configuration.create_save_filenames_for_text_file_paths()

        output_paths = []
        for text, filename in zip(texts, filenames):
            wav, sample_rate = self.generate(text)
            output_paths.append(self.save_output(wav, sample_rate, filename))

        return output_paths

