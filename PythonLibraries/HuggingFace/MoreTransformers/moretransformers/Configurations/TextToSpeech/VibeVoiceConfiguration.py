from pydantic import BaseModel, Field, field_validator
from pathlib import Path
from typing import List, Optional,Union

class VibeVoiceConfiguration(BaseModel):
    audio_file_paths: List[Union[str, Path]] = Field(
        default_factory=list,
        description="List of paths to .wav audio files"
    )
    
    text_file_paths: List[Union[str, Path]] = Field(
        default_factory=list,
        description="List of paths to text files"
    )

    cfg_scale: float = Field(
        default=1.3,
        description="CFG scale"
    )

    max_new_tokens: Optional[int] = Field(
        default=None,
        description="Max new tokens"
    )

    @field_validator('audio_file_paths')
    @classmethod
    def validate_audio_files(cls, v):
        """Validate that all audio file paths exist and are .wav files."""
        if not v:
            return v
        
        validated_paths = []
        for audio_path in v:
            path = Path(audio_path)
            if not path.exists():
                raise ValueError(f"Audio file not found: {audio_path}")
            if path.suffix.lower() != '.wav':
                raise ValueError(
                    f"Audio file must be .wav format: {audio_path}")
            validated_paths.append(path)

        return validated_paths

    @field_validator('text_file_paths')
    @classmethod
    def validate_text_files(cls, v):
        """Validate that all text file paths exist."""
        if not v:
            return v
        
        validated_paths = []
        for text_path in v:
            path = Path(text_path)
            if not path.exists():
                raise ValueError(f"Text file not found: {text_path}")
            validated_paths.append(path)
        
        return validated_paths
    
    def get_audio_file_paths(self) -> List[str]:
        """Get all audio file paths as strings."""
        return [str(path) for path in self.audio_file_paths]

    def get_text_file_paths(self) -> List[str]:
        """Get all text file paths as strings."""
        return [str(path) for path in self.text_file_paths]
