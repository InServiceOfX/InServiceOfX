from clitexttospeech.Configuration import CLIConfiguration
from pathlib import Path

test_data_path = Path(__file__).resolve().parents[2] / "TestData"

def test_CLIConfiguration_inits():
    cli_configuration = CLIConfiguration()
    assert cli_configuration.text_file_path is None
    assert cli_configuration.text_file_paths is None

def test_CLIConfiguration_from_yaml_single_text_file():
    cli_configuration = CLIConfiguration.from_yaml(
        test_data_path / "single_file_cli_configuration.yml")
    assert cli_configuration.text_file_path == Path("dia_text.txt")
    assert cli_configuration.text_file_paths is None
