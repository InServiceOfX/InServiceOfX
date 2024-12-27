import pytest
from clichat.Chatbot import Chatbot
from clichat.Configuration import Configuration
from prompt_toolkit.styles import Style
from pathlib import Path
from prompt_toolkit.document import Document
from prompt_toolkit.completion import FuzzyCompleter, WordCompleter
from prompt_toolkit.completion.base import CompleteEvent
from clichat.Configuration import RuntimeConfiguration

test_data_directory = Path(__file__).resolve().parents[1] / "TestData"

@pytest.fixture
def config():
    test_file_path = test_data_directory / "clichat_configuration.yml"
    assert test_file_path.exists()
    return Configuration(test_file_path)

def test_chatbot_init_with_defaults():
    # Test initialization with default values
    chatbot = Chatbot()
    
    assert chatbot.temperature == 1.0
    assert isinstance(chatbot.prompt_style, Style)
    
    # Check default style values
    style_dict = dict(chatbot.prompt_style._style_rules)
    assert style_dict.get("") == "ansigreen"  # Default text color
    assert style_dict.get("indicator") == "ansicyan"  # Prompt indicator color

def test_chatbot_init_with_configuration(config):
    # Test initialization with configuration object
    chatbot = Chatbot(configuration=config)
    
    assert chatbot.temperature == config.temperature
    assert isinstance(chatbot.prompt_style, Style)
    
    # Check configured style values
    style_dict = dict(chatbot.prompt_style._style_rules)
    assert style_dict.get("") == config.terminal_CommandEntryColor2
    assert style_dict.get("indicator") == config.terminal_PromptIndicatorColor2

def test_chatbot_init_with_custom_name():
    # Test initialization with custom name
    custom_name = "CustomBot"
    chatbot = Chatbot(name=custom_name)
    
    assert chatbot.temperature == 1.0  # Default value
    assert isinstance(chatbot.prompt_style, Style)

def test_chatbot_init_with_none_configuration_values(config):
    # Test handling of None values in configuration
    config.temperature = None
    config.terminal_CommandEntryColor2 = None
    config.terminal_PromptIndicatorColor2 = None
    
    chatbot = Chatbot(configuration=config)
    
    # Should use default values when configuration values are None
    assert chatbot.temperature == 1.0
    
    # Check default style values are used when configuration values are None
    style_dict = dict(chatbot.prompt_style._style_rules)
    assert style_dict.get("") == "ansigreen"
    assert style_dict.get("indicator") == "ansicyan"

def test_create_completer_returns_none_when_messages_exist(config):
    chatbot = Chatbot(configuration=config)
    runtime_config = RuntimeConfiguration()
    runtime_config.current_messages = [{"role": "user", "content": "Hello"}]
    
    completer = chatbot._create_completer(runtime_config)
    assert completer is None

def test_create_completer_returns_fuzzy_completer_when_no_messages(
    config):
    chatbot = Chatbot(configuration=config)
    runtime_config = RuntimeConfiguration()
    runtime_config.current_messages = None
    
    completer = chatbot._create_completer(runtime_config)
    assert completer is not None
    assert isinstance(completer, FuzzyCompleter)
    
    # Get the wrapped completer
    word_completer = completer.completer
    assert isinstance(word_completer, WordCompleter)
    
    # Test the word completer's properties
    assert word_completer.ignore_case == True
    
    # Verify all expected commands are in the completer
    expected_commands = {
        ".model",
        ".active_system_messages",
        ".add_system_message",
        ".configure_system_messages",
        ".temperature",
        ".togglewordwrap",
        config.exit_entry
    }
    assert set(word_completer.words) == expected_commands

def test_create_completer_completions_work(config):
    chatbot = Chatbot(configuration=config)
    runtime_config = RuntimeConfiguration()
    runtime_config.current_messages = None
    
    completer = chatbot._create_completer(runtime_config)
    
    # Test some completion scenarios
    document = Document(".activ")  # Simulating typing .sys
    completions = list(completer.get_completions(document, CompleteEvent()))
    assert any(c.text == ".active_system_messages" for c in completions)
    
    document = Document(".temp")  # Simulating typing .temp
    completions = list(completer.get_completions(document, CompleteEvent()))
    assert any(c.text == ".temperature" for c in completions)