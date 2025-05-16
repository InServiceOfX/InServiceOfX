from moresglang.Wrappers.OpenAI import Client

from commonapi.Messages import (
    ConversationHistoryAndSystemMessagesManager,
    UserMessage,
)

class ClientAndMessagesManager:
    def __init__(
            self,
            server_configuration,
            openai_chat_completion_configuration):
        self.server_configuration = server_configuration
        self.openai_chat_completion_configuration = \
            openai_chat_completion_configuration

        self.client = Client(
            server_configuration,
            openai_chat_completion_configuration)

        # chsmm = conversation history and system messages manager
        self.chsmm = ConversationHistoryAndSystemMessagesManager()

    def generate_from_single_user_content(self, user_content):
        user_message = UserMessage(content=user_content)
        self.chsmm.conversation_history.append_message(user_message)

        response = self.client.create_chat_completion(
            self.chsmm.conversation_history.as_list_of_dicts())

        parsed_completion = Client.get_parsed_completion(response)

        self.chsmm.conversation_history.append_message(
            AssistantMessage(content=parsed_completion["content"]))

        return parsed_completion["content"]