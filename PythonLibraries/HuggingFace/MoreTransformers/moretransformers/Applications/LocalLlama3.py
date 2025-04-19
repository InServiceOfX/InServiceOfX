from moretransformers.Wrappers.LLMEngines import LocalLlama as LocalLlamaEngine

from commonapi.Messages import (
    AssistantMessage,
    ConversationHistory,
    SystemMessage,
    SystemMessagesManager,
    UserMessage,
)

class LocalLlama3:
    def __init__(self, configuration, generation_configuration):
        self.configuration = configuration
        self.generation_configuration = generation_configuration

        self.llm_engine = LocalLlamaEngine(
            configuration,
            generation_configuration)

        self.conversation_history = ConversationHistory()
        self.system_messages_manager = SystemMessagesManager()

    def clear_conversation_history(self, is_keep_active_system_messages=True):
        self.conversation_history.clear()
        if is_keep_active_system_messages:
            active_system_messages = \
                self.system_messages_manager.get_active_messages()
            for message in active_system_messages:
                self.conversation_history.append_message(
                    SystemMessage(message.content))

    def add_system_message(
            self,
            system_message_content,
            is_clear_conversation_history=False):
        if is_clear_conversation_history:
            self.clear_conversation_history()

        if system_message_content is None or system_message_content == "":
            return None

        # Make system message active since we will be appending it to
        # conversation history.
        add_message_result = self.system_messages_manager.add_message(
            system_message_content,
            True)

        if add_message_result is None:
            return None

        self.conversation_history.append_message(
            SystemMessage(system_message_content))

        return add_message_result

    def generate_from_single_user_content(self, user_content):
        user_message = UserMessage(content=user_content)
        self.conversation_history.append_message(user_message)

        response = self.llm_engine.generate_for_llm_engine(
            self.conversation_history.as_list_of_dicts())
        
        self.conversation_history.append_message(
            AssistantMessage(content=response))

        return response
