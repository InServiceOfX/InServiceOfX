from pydantic import BaseModel, Field, field_validator
from pathlib import Path
from typing import List, Optional, Tuple, Union

import hashlib, json, time, yaml

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

    directory_path_to_save: Union[str, Path] = Field(
        default=None,
        description="Directory path to save the generated speech"
    )

    base_saved_filename: Optional[str] = Field(
        default="VibeVoiceOutput",
        description="Base filename to save the generated speech"
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

    @classmethod
    def from_yaml(cls, file_path: str | Path) -> "VibeVoiceConfiguration":
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def get_audio_file_paths(self) -> List[str]:
        """Get all audio file paths as strings."""
        return [str(path) for path in self.audio_file_paths]

    def get_text_file_paths(self) -> List[str]:
        """Get all text file paths as strings."""
        return [str(path) for path in self.text_file_paths]

    def _create_configuration_hash(self) -> Tuple[str, str]:
        current_timestamp = time.time()
        config_dict = {
            "timestamp": current_timestamp,
            "audio_file_paths": self.audio_file_paths,
            "text_file_paths": self.text_file_paths,
            "cfg_scale": self.cfg_scale,
            "max_new_tokens": self.max_new_tokens,
        }
        config_json = json.dumps(config_dict, sort_keys=True, default=str)
        hash_object = hashlib.sha256(config_json.encode('utf-8'))
        full_hash = hash_object.hexdigest()
        truncated_hash = full_hash[:12]
        return full_hash, truncated_hash

    def create_save_filename(self) -> str:
        full_hash, truncated_hash = self._create_configuration_hash()
        filename = f"{self.base_saved_filename}-{truncated_hash}.wav"
        return filename, full_hash