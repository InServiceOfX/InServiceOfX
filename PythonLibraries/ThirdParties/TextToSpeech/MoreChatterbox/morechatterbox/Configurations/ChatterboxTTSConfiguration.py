from pathlib import Path
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Tuple, Union

import hashlib
import json
import time
import yaml


class ChatterboxTTSConfiguration(BaseModel):
    model_dir: Union[str, Path] = Field(
        ...,
        description="Path to the Chatterbox checkpoint directory",
    )

    device: Optional[str] = Field(
        default="cuda:0",
        description="Device to run the model on (e.g. cuda:0)",
    )

    audio_prompt_path: Union[str, Path] = Field(
        ...,
        description="Path to a single .wav audio file used as the voice prompt",
    )

    text_file_path: Union[str, Path] = Field(
        ...,
        description="Path to the single text file to synthesize",
    )

    text_file_paths: Optional[List[Union[str, Path]]] = Field(
        default=None,
        description="List of paths to text files to synthesize",
    )

    directory_path_to_save: Union[str, Path] = Field(
        ...,
        description="Directory path to save the generated speech",
    )

    base_saved_filename: Optional[str] = Field(
        default="ChatterboxOutput",
        description="Base filename for the saved audio (hash will be appended)",
    )

    sample_rate: Optional[int] = Field(
        default=None,
        description="Sample rate for saved audio; if None, use model default",
    )

    @field_validator("model_dir")
    @classmethod
    def validate_model_dir(cls, v):
        path = Path(v)
        if not path.exists():
            raise ValueError(f"Model directory not found: {v}")
        if not path.is_dir():
            raise ValueError(f"Model path is not a directory: {v}")
        return path

    @field_validator("audio_prompt_path")
    @classmethod
    def validate_audio_prompt_path(cls, v):
        path = Path(v)
        if not path.exists():
            raise ValueError(f"Audio prompt file not found: {v}")
        if path.suffix.lower() != ".wav":
            raise ValueError(f"Audio prompt must be .wav: {v}")
        return path

    @field_validator("text_file_path")
    @classmethod
    def validate_text_file_path(cls, v):
        path = Path(v)
        if not path.exists():
            raise ValueError(f"Text file not found: {v}")
        return path

    @field_validator("text_file_paths")
    @classmethod
    def validate_text_file_paths(cls, v):
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
    def from_yaml(cls, file_path: Union[str, Path]) -> "ChatterboxTTSConfiguration":
        with open(file_path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def get_audio_prompt_path(self) -> Path:
        return Path(self.audio_prompt_path)

    def get_text_file_path(self) -> Path:
        return Path(self.text_file_path)

    def get_text_file_paths(self) -> List[Path]:
        return [Path(path) for path in self.text_file_paths]

    def _create_configuration_hash(
        self,
        text_file_path: Optional[Path] = None) -> Tuple[str, str]:
        current_timestamp = time.time()
        config_dict = {
            "timestamp": current_timestamp,
            "model_dir": str(self.model_dir),
            "audio_prompt_path": str(self.audio_prompt_path),
            "text_file_path": str(self.text_file_path) \
                if text_file_path is None else str(text_file_path),
        }
        config_json = json.dumps(config_dict, sort_keys=True, default=str)
        hash_object = hashlib.sha256(config_json.encode("utf-8"))
        full_hash = hash_object.hexdigest()
        truncated_hash = full_hash[:12]
        return full_hash, truncated_hash

    def _create_configuration_hashes_for_text_file_paths(self) -> \
        List[Tuple[str, str]]:
        hashes = []
        for text_file_path in self.text_file_paths:
            hashes.append(self._create_configuration_hash(text_file_path))
        return hashes

    def create_save_filename(self) -> Tuple[str, str]:
        full_hash, truncated_hash = self._create_configuration_hash()
        filename = f"{self.base_saved_filename}-{truncated_hash}.wav"
        return filename, full_hash

    def create_save_filenames_for_text_file_paths(self) -> \
        List[Tuple[str, str]]:
        hashes = self._create_configuration_hashes_for_text_file_paths()
        filenames = []
        for index, (_, truncated_hash) in enumerate(hashes):
            filenames.append(
                f"{self.base_saved_filename}-{index}-{truncated_hash}.wav")
        return filenames