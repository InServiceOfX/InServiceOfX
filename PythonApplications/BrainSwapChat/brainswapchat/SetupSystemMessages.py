from brainswapchat import ApplicationPaths
from commonapi.Messages import SystemMessagesManager
from commonapi.FileIO import SystemMessagesFileIO

def setup_system_messages(
        application_paths: ApplicationPaths,
        conversation_and_system_messages):

    path_field_name = "system_messages_file_path"

    result = application_paths.create_missing_files(path_field_name)

    if not result[path_field_name]:
        raise RuntimeError(
            f"Failed to create system messages file at {path_field_name}.")

    system_messages_file_io = SystemMessagesFileIO(
        application_paths.system_messages_file_path)

    if not system_messages_file_io.load_messages() and \
            system_messages_file_io.messages == None:
        raise RuntimeError(
            f"Failed to load system messages from {path_field_name}.")

    system_messages_file_io.put_messages_into_system_messages_manager(
        conversation_and_system_messages.system_messages_manager)

    system_messages_file_io.put_messages_into_conversation_history(
        conversation_and_system_messages.conversation_history)