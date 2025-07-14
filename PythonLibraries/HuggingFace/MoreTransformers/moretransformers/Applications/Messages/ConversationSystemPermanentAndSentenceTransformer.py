from commonapi.Messages import ConversationSystemAndPermanent
from commonapi.Messages.Messages import AssistantMessage, Message
from warnings import warn
from typing import Any, List, Dict
from .MakeMessageEmbeddingsWithSentenceTransformer import \
    MakeMessageEmbeddingsWithSentenceTransformer

class ConversationSystemPermanentAndSentenceTransformer:
    def __init__(self, embedding_model):
        self.csp = ConversationSystemAndPermanent()
        self.mmewst = \
            MakeMessageEmbeddingsWithSentenceTransformer(embedding_model)

    def clear_conversation_history(self, is_keep_active_system_messages=True):
        self.csp.clear_conversation_history(
            is_keep_active_system_messages=is_keep_active_system_messages)

    def add_system_message(
            self,
            system_message_content,
            is_clear_conversation_history=False):
        add_system_message_result = \
            self.csp.conversation_and_system_messages.add_system_message(
                system_message_content,
                is_clear_conversation_history=is_clear_conversation_history)

        embedding = self.mmewst.make_embedding_from_content(
            content=system_message_content,
            role="system")

        self.csp.permanent_conversation.add_message_as_content(
            content=system_message_content,
            role="system",
            embedding=embedding)

    def append_message(self, message: Message):
        self.csp.conversation_and_system_messages.append_message(
            message)

        embedding = self.mmewst.make_embedding_from_message(
            message)

        self.csp.permanent_conversation.add_message(
            message,
            embedding=embedding)

        if message.role == "user":
            self.csp._current_user_message = \
                message
        elif message.role == "assistant":
            self.csp._current_assistant_message = \
                message
            if self.csp._current_user_message \
                is not None:
                embedding = \
                    self.mmewst.make_embedding_from_message_pair(
                        self.csp._current_user_message,
                        message)
                self.csp.permanent_conversation.append_message_pair(
                    message_0=\
                        self.csp._current_user_message,
                    message_1=\
                        self.csp._current_assistant_message,
                    embedding=embedding)
                self.csp._current_user_message = None
                self.csp._current_assistant_message = None


    def append_general_message(self, message: Any):
        self.csp.append_general_message(message)

    def handle_groq_chat_completion_response(self, response: Any) -> None:
        if response and hasattr(response, "choices") and \
            len(response.choices) > 0:
            assistant_message_content = response.choices[0].message
            self.append_message(AssistantMessage(assistant_message_content))
        else:
            warn("No assistant message content in response.")
            self.append_general_message(response)

    def get_conversation_as_list_of_dicts(self) -> List[Dict[str, Any]]:
        return \
            self.csp.get_conversation_as_list_of_dicts()

    def get_permanent_conversation_messages(self):
        return \
            self.csp.get_permanent_conversation_messages()

    def get_permanent_conversation_message_pairs(self):
        return \
            self.csp.get_permanent_conversation_message_pairs()