from pathlib import Path
import pytest
import tempfile
import yaml

from morechatterbox.Configurations import ChatterboxTTSConfiguration

def test_ChatterboxTTSConfiguration_from_yaml_string_and_temp_dir():
    """Config is built from a YAML string; paths point into a temp dir that is
    removed after the test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # Paths that validators require to exist
        model_dir = tmp / "model"
        model_dir.mkdir()
        audio_prompt_path = tmp / "prompt.wav"
        audio_prompt_path.write_bytes(b"")  # existence + .wav suffix is enough for validators
        text_file_path = tmp / "script.txt"
        text_file_path.write_text("Hello world.")
        directory_path_to_save = tmp / "output"
        directory_path_to_save.mkdir()

        # YAML as string with temp paths injected (no TestData file)
        yaml_str = f"""
model_dir: {model_dir}
device: cuda:0
audio_prompt_path: {audio_prompt_path}
text_file_path: {text_file_path}
directory_path_to_save: {directory_path_to_save}
base_saved_filename: ChatterboxTest
"""
        # Write to a temp config file so we exercise from_yaml
        config_path = tmp / "config.yml"
        config_path.write_text(yaml_str)

        configuration = ChatterboxTTSConfiguration.from_yaml(config_path)

        assert configuration is not None
        assert configuration.model_dir == model_dir
        assert configuration.device == "cuda:0"
        assert configuration.get_audio_prompt_path() == audio_prompt_path
        assert configuration.get_text_file_path() == text_file_path
        assert configuration.directory_path_to_save == str(
            directory_path_to_save)
        assert configuration.base_saved_filename == "ChatterboxTest"

        filename, full_hash = configuration.create_save_filename()
        assert filename.endswith(".wav")
        assert "ChatterboxTest" in filename
        # sha256 hex is 64 characters
        assert len(full_hash) == 64

    # After the `with` block, tmpdir is gone; nothing left on disk.