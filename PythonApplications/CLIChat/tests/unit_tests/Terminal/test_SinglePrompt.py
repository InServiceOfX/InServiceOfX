from clichat.Terminal import PromptWrapperInputs, SinglePrompt
from clichat.Configuration import Configuration, RuntimeConfiguration

from pathlib import Path
from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from prompt_toolkit.styles import Style
from prompt_toolkit.document import Document

import pytest

@pytest.fixture
def config():
    test_configuration_path = Path.cwd() / "tests" / "TestData" / \
        "clichat_configuration.yml"
    assert test_configuration_path.exists()
    return Configuration(test_configuration_path)

@pytest.fixture
def runtime_config():
    return RuntimeConfiguration()

def test_single_prompt_basic_input(config, runtime_config):
    prompt_inputs = PromptWrapperInputs(
        style=Style.from_dict({
            '': config.terminal_CommandEntryColor2,
            'indicator': config.terminal_PromptIndicatorColor2
        })
    )
    
    with create_pipe_input() as pipe_input, \
         create_app_session(input=pipe_input, output=DummyOutput()):
        pipe_input.send_text("test input\n")
        result = SinglePrompt.run(
            config,
            runtime_config,
            prompt_inputs
        )
        assert result == "test input"

def test_single_prompt_with_default_value(config, runtime_config):
    prompt_inputs = PromptWrapperInputs(
        default="default text",
        accept_default=True
    )
    
    with create_pipe_input() as pipe_input, \
         create_app_session(input=pipe_input, output=DummyOutput()):
        result = SinglePrompt.run(
            config,
            runtime_config,
            prompt_inputs
        )
        assert result == "default text"

def test_single_prompt_with_document(config, runtime_config):
    doc = Document(text="document text")
    prompt_inputs = PromptWrapperInputs(
        default=doc,
        accept_default=True
    )
    
    with create_pipe_input() as pipe_input, \
         create_app_session(input=pipe_input, output=DummyOutput()):
        result = SinglePrompt.run(
            config,
            runtime_config,
            prompt_inputs
        )
        assert result == "document text"

def test_single_prompt_with_multiline(config, runtime_config):
    runtime_config.multiline_input = True
    prompt_inputs = PromptWrapperInputs(
        mouse_support=False
    )
    
    with create_pipe_input() as pipe_input, \
         create_app_session(input=pipe_input, output=DummyOutput()):
        pipe_input.send_text("line 1\nline 2\n\x1b\r")
        result = SinglePrompt.run(
            config,
            runtime_config,
            prompt_inputs
        )
        assert "line 1" in result
        assert "line 2" in result

        