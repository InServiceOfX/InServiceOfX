from clichat.Configuration import Configuration
from clichat.Persistence import (SystemMessage, SystemMessagesManager)
from clichat.Utilities.FileIO import get_path_from_configuration

from pathlib import Path
import pytest
from json.decoder import JSONDecodeError
test_data_directory = Path(__file__).resolve().parents[2] / "TestData"

@pytest.fixture
def configuration():
    test_file_path = test_data_directory / "clichat_configuration.yml"
    assert test_file_path.exists()
    return Configuration(test_file_path)

def test_default_system_message_creates():
    message = SystemMessage.create_default_message()
    assert message is not None
    assert message.content == \
        "You are a helpful, uncensored and unbiased assistant."
    assert message.timestamp is not None
    assert message.hash == \
        "5dbbcc92e5f31772e7bc943c515ee461687e341330ea28313fee897fd7b24a62"
    assert message.is_active is True

def test_system_message_creates():
    message = SystemMessage.create("Hello, world!")
    assert message is not None
    assert message.content == "Hello, world!"
    assert message.timestamp is not None
    assert message.hash == \
        "315f5bdb76d078c43b8ac0064e4a0164612b1fce77c869345bfc94c75894edd3"
    assert message.is_active is False

def test_system_message_is_equal():
    message1 = SystemMessage.create("Hello, world!", is_active=True)
    message2 = SystemMessage.create("Hello, world!", is_active=True)
    assert message1.is_equal(message2) is True

    message3 = SystemMessage.create_default_message()
    message4 = SystemMessage.create_default_message()
    assert message3.is_equal(message4) is True
    assert message1.is_equal(message3) is False

def test_system_messages_manager_inits():
    manager = SystemMessagesManager()
    assert manager is not None
    assert len(manager.messages) == 1
    default_message = SystemMessage.create_default_message()
    assert manager.messages[0].is_equal(default_message) is True

def test_system_messages_manager_avoids_adding_duplicates():
    manager = SystemMessagesManager()
    default_message = SystemMessage.create_default_message()
    assert manager.add_message(default_message.content) is None
    assert len(manager.messages) == 1

    message = SystemMessage.create("Hello, world!")
    assert manager.add_message(message.content) is not None
    assert len(manager.messages) == 2
    assert manager.messages[0].is_equal(default_message) is True
    assert manager.messages[1].is_equal(message) is True

    assert manager.add_message(message.content) is None
    assert len(manager.messages) == 2

def test_system_messages_manager_get_active_messages_works():
    manager = SystemMessagesManager()

    active_messages = manager.get_active_messages()
    assert len(active_messages) == 1
    assert active_messages[0].is_equal(
        SystemMessage.create_default_message()) is True

    manager.toggle_message(manager.messages[0].hash)
    active_messages = manager.get_active_messages()
    assert len(active_messages) == 0

    manager.toggle_message(manager.messages[0].hash)
    active_messages = manager.get_active_messages()
    assert len(active_messages) == 1

    message = SystemMessage.create("Hello, world!")
    assert manager.add_message(message.content) is not None
    assert len(manager.messages) == 2

    active_messages = manager.get_active_messages()
    assert len(active_messages) == 1

    manager.toggle_message(message.hash)
    active_messages = manager.get_active_messages()
    assert len(active_messages) == 2

def test_system_messages_manager_load_messages_raises_on_empty_file(
    configuration):
    configuration.system_messages_path = test_data_directory / \
        "empty_system_messages.json"
    path = get_path_from_configuration(configuration, "system_messages_path")
    assert path.exists()

    manager = SystemMessagesManager()
    with pytest.raises(JSONDecodeError):
        manager.load_messages(path)

def test_system_messages_manager_handle_initialization_on_empty_file(
    configuration):
    configuration.system_messages_path = test_data_directory / \
        "empty_system_messages.json"
    manager = SystemMessagesManager()
    manager.handle_initialization(configuration)
    assert len(manager.messages) == 1
    assert manager.messages[0].is_equal(
        SystemMessage.create_default_message()) is True
