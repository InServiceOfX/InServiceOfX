from moretransformers.Wrappers.LLMEngines import LocalLlama as LocalLlamaEngine

from commonapi.Messages import (
    AssistantMessage,
    ConversationHistory,
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

    def clear_conversation_history(self):
        self.conversation_history.clear()

    def generate(self, messages):
        for message in messages:
            self.conversation_history.append_message(
                parse_dict_into_specific_message(message))

        response = self.llm_engine.generate(messages)

        self.conversation_history.append_message(
            AssistantMessage(content=response))

        return response
