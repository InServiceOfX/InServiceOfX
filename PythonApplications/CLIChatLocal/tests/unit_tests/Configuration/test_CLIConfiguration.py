from clichatlocal.Configuration import CLIConfiguration

from pathlib import Path

def test_CLIConfiguration_inits_with_default_values():
    config = CLIConfiguration()
    assert config.exit_command == ".exit"
    assert config.help_command == ".help"
    assert config.user_color == "ansigreen"
    assert config.assistant_color == "ansiblue"
    assert config.system_color == "ansiyellow"
    assert config.info_color == "ansicyan"
    assert config.error_color == "ansired"
