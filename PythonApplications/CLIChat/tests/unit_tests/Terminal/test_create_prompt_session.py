from clichat.Configuration import Configuration
from clichat.Terminal import create_prompt_session
from clichat.Utilities.FileIO import get_existing_chat_history_path_or_fail
from pathlib import Path
from prompt_toolkit.history import FileHistory
from prompt_toolkit import PromptSession, prompt
from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

def test_create_prompt_session_steps():
    test_configuration_path = Path.cwd() / "tests" / "TestData" / \
        "clichat_configuration.yml"
    assert test_configuration_path.exists()
    
    configuration = Configuration(test_configuration_path)

    chat_history_path = get_existing_chat_history_path_or_fail(
        configuration)
    assert chat_history_path.exists()

    # https://python-prompt-toolkit.readthedocs.io/en/master/pages/reference.html#prompt_toolkit.history.FileHistory
    # FileHistory is a History class that stores all strings in a file.
    history = FileHistory(chat_history_path)
    
    with create_pipe_input() as pipe_input, \
         create_app_session(input=pipe_input, output=DummyOutput()):
        chat_session = PromptSession(history=history)
        pipe_input.send_text("test input\n")
        text1 = chat_session.prompt()
        assert text1 == "test input"

def test_create_prompt_session_works():
    test_configuration_path = Path.cwd() / "tests" / "TestData" / \
        "clichat_configuration.yml"
    configuration = Configuration(test_configuration_path)
    chat_history_path = get_existing_chat_history_path_or_fail(
        configuration)

    with create_pipe_input() as pipe_input, \
         create_app_session(input=pipe_input, output=DummyOutput()):
        prompt_session = create_prompt_session(chat_history_path)
        pipe_input.send_text("test input 2\n")
        text1 = prompt_session.prompt()
        assert text1 == "test input 2"

def test_prompt_session_provides_prompt():
    test_configuration_path = Path.cwd() / "tests" / "TestData" / \
        "clichat_configuration.yml"
    configuration = Configuration(test_configuration_path)
    chat_history_path = get_existing_chat_history_path_or_fail(
        configuration)

    with create_pipe_input() as pipe_input, \
         create_app_session(input=pipe_input, output=DummyOutput()):
        prompt_session = create_prompt_session(chat_history_path)
        
        # Test both prompt methods with the same input
        pipe_input.send_text("test input 3\n")
        result1 = prompt_session.prompt()
        
        pipe_input.send_text("test input 3\n")
        result2 = prompt()
        
        assert result1 == result2 == "test input 3"


