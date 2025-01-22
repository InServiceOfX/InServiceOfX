import pytest
from corecode.Utilities import load_environment_file
from unittest.mock import Mock, patch, AsyncMock
from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from clichat.Terminal.GroqModelSelector import GroqModelSelector
from clichat.Configuration import Configuration
from dataclasses import dataclass

load_environment_file()

@dataclass
class MockModel:
    id: str
    context_window: int

@pytest.fixture
def config():
    config = Mock()
    config.terminal_DialogBackgroundColor = "black"
    return config

@pytest.fixture
def model_selector(config):
    return GroqModelSelector(config)

def test_model_selector_initialization(config):
    selector = GroqModelSelector(config)
    assert selector.style is not None
    style_dict = dict(selector.style._style_rules)
    assert style_dict['dialog'] == 'bg:black fg:white'
    assert style_dict['dialog.body'] == 'bg:black fg:white'
    assert style_dict['dialog frame.label'] == 'bg:black fg:white'

@patch('moregroq.Wrappers.GetAllActiveModels')
def test_get_available_models_gets_default_model(mock_get_models, model_selector):
    mock_models = [
        MockModel(id="model-1", context_window=1024),
        MockModel(id="model-2", context_window=2048)
    ]
    
    mock_get_models_instance = Mock()
    mock_get_models_instance.get_list_of_available_models.return_value = \
        mock_models
    mock_get_models.return_value = mock_get_models_instance

    models = model_selector._get_available_models()
    assert len(models) == 16

@patch('moregroq.Wrappers.GetAllActiveModels')
def test_get_available_models_failure_gets_default_model(
    mock_get_models,
    model_selector):
    mock_get_models.side_effect = Exception("API Error")
    
    models = model_selector._get_available_models()
    assert len(models) == 16
