from moretransformers.Wrappers.LLMEngines import LocalLlama as LocalLlamaEngine

from commonapi.Messages import (
    AssistantMessage,
    ConversationHistory,
    SystemMessage,
    UserMessage,
    parse_dict_into_specific_message
)

class LocalLlama3:
    def __init__(self, configuration, generation_configuration):
        self.configuration = configuration
        self.generation_configuration = generation_configuration

        self.llm_engine = LocalLlamaEngine(
            configuration,
            generation_configuration)

        self.conversation_history = ConversationHistory()

        self.current_system_message = None

    def clear_conversation_history(self):
        self.conversation_history.clear()

    def set_system_message(
            self,
            system_message_content,
            is_clear_conversation_history=False):
        if is_clear_conversation_history:
            self.clear_conversation_history()

        self.current_system_message = SystemMessage(
            content=system_message_content)
        
        self.conversation_history.append_message(self.current_system_message)

    def generate_from_single_user_content(self, user_content):
        user_message = UserMessage(content=user_content)
        self.conversation_history.append_message(user_message)

        response = self.llm_engine.generate_for_llm_engine(
            self.conversation_history.as_list_of_dicts())
        
        self.conversation_history.append_message(
            AssistantMessage(content=response))

        return response
