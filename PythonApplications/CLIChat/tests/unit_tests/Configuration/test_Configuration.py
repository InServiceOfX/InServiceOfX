from clichat.Configuration import Configuration
from pathlib import Path

test_data_directory = Path(__file__).resolve().parents[2] / "TestData"

def test_Configuration_loads_from_yaml():
    test_file_path = test_data_directory / "clichat_configuration.yml"
    assert test_file_path.exists()
    
    configuration = Configuration(test_file_path)
    
    # Test that values from yaml are loaded correctly
    assert configuration.temperature == 1.0
    assert configuration.terminal_DisplayCommandOnMenu == False
    assert configuration.terminal_CommandEntryColor2 == "ansigreen"
    assert configuration.terminal_PromptIndicatorColor2 == "ansicyan"
    assert configuration.terminal_ResourceLinkColor == "ansiyellow"

