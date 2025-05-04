from clichatlocal.Configuration import CLIConfiguration

from pathlib import Path

test_data_directory = Path(__file__).resolve().parents[2] / "TestData"

def test_CLIConfiguration_inits_with_minimal_yaml_file():
    config = CLIConfiguration.from_yaml(
        test_data_directory / "cli_configuration.yml",
        is_dev=True)
    assert config.inference_mode == "transformers"

    assert config.exit_command == ".exit"
    assert config.help_command == ".help"

    assert config.user_color == "ansigreen"
    assert config.assistant_color == "ansiblue"
    assert config.system_color == "ansiyellow"
    assert config.info_color == "ansicyan"
    assert config.error_color == "ansired"

    assert str(config.file_history_path) == str(Path(__file__).resolve().parents[3] / \
        "Configurations" / "file_history.txt")
    assert config.terminal_CommandEntryColor2 == "ansigreen"
    assert config.terminal_PromptIndicatorColor2 == "ansicyan"

