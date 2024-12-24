import pytest
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.application import Application
from unittest.mock import Mock, patch
from clichat.Configuration import Configuration, RuntimeConfiguration
from clichat.Utilities import ConfigureKeyBindings
from pathlib import Path

test_data_directory = Path(__file__).resolve().parents[2] / "TestData"

@pytest.fixture
def config():
    test_file_path = test_data_directory / "clichat_configuration.yml"
    assert test_file_path.exists()
    return Configuration(test_file_path)

@pytest.fixture
def runtime_config():
    return RuntimeConfiguration()

@pytest.fixture
def mock_event():
    buffer = Mock(spec=Buffer)
    app = Mock(spec=Application)
    app.current_buffer = buffer
    event = Mock()
    event.app = app
    return event

@pytest.fixture
def key_binder(config, runtime_config):
    return ConfigureKeyBindings(config, runtime_config)

def test_exit_binding(key_binder, mock_event):
    kb = key_binder.configure_key_bindings()
    
    # Find and execute exit binding
    for binding in kb.bindings:
        if binding.keys == tuple(key_binder._configuration.hotkey_exit):
            binding.handler(mock_event)
            break
    
    mock_event.app.current_buffer.text = key_binder._configuration.exit_entry
    mock_event.app.current_buffer.validate_and_handle.assert_called_once()

def test_cancel_binding(key_binder, mock_event):
    kb = key_binder.configure_key_bindings()
    
    # Find and execute cancel binding
    for binding in kb.bindings:
        if binding.keys == tuple(key_binder._configuration.hotkey_cancel):
            binding.handler(mock_event)
            break
    
    mock_event.app.current_buffer.reset.assert_called_once()

def test_newline_binding(key_binder, mock_event):
    kb = key_binder.configure_key_bindings()
    
    # Find and execute newline binding
    for binding in kb.bindings:
        if binding.keys == tuple(
            key_binder._configuration.hotkey_insert_newline):
            binding.handler(mock_event)
            break
    
    mock_event.app.current_buffer.newline.assert_called_once()


def test_new_binding(key_binder, mock_event):
    kb = key_binder.configure_key_bindings()
    mock_event.app.current_buffer.text = "existing text"
    
    # Find and execute new binding
    for binding in kb.bindings:
        if binding.keys == tuple(key_binder._configuration.hotkey_new):
            binding.handler(mock_event)
            break
    
    assert key_binder._default_entry == "existing text"
    assert mock_event.app.current_buffer.text == ".new"
    mock_event.app.current_buffer.validate_and_handle.assert_called_once()