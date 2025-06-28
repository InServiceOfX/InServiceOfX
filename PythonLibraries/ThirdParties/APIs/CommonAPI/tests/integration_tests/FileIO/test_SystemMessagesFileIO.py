from commonapi.FileIO import SystemMessagesFileIO
from commonapi.Messages import (
    RecordedSystemMessage,
    SystemMessage,
    SystemMessagesManager)

from pathlib import Path

test_data_path = Path(__file__).parents[2] / "TestData"
test_file_path = test_data_path / "test_system_messages.json"

def test_save_and_load_messages():
    test_data_path.mkdir(parents=True, exist_ok=True)

    test_file_path.unlink(missing_ok=True)

    test_file_path.touch()

    file_io = SystemMessagesFileIO(test_file_path)

    assert file_io.is_file_path_valid()

    system_messages_manager = SystemMessagesManager()
    system_messages_manager.clear()
    system_messages_manager.add_message("Hello, world!")

    default_message = RecordedSystemMessage.create_default_message().content
    system_messages_manager.add_message(default_message)

    data = [msg.__dict__ for msg in system_messages_manager.messages]
    for element in data:
        print("element: ", element)

    save_result = file_io.save_messages(system_messages_manager.messages)
    print("save_result", save_result)
    file_io.load_messages()

    for key, value in file_io.messages.items():
        assert key in system_messages_manager._messages_dict

    # Clean up.
    test_file_path.unlink(missing_ok=True)