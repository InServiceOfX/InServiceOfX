from .PermanentConversation import PermanentConversation
from commonapi.Messages import (
    AssistantMessage,
    ConversationAndSystemMessages,
    SystemMessage)
from commonapi.Messages.Messages import Message
from typing import List, Dict, Any
from warnings import warn

class ConversationSystemAndPermanent:
    def __init__(self):
        self.conversation_and_system_messages = ConversationAndSystemMessages()
        self.permanent_conversation = PermanentConversation()
        self._current_user_message = None
        self._current_assistant_message = None

    def clear_conversation_history(self, is_keep_active_system_messages=True):
        self.conversation_and_system_messages.clear_conversation_history(
            is_keep_active_system_messages=is_keep_active_system_messages)

        if is_keep_active_system_messages:
            active_system_messages = \
                self.conversation_and_system_messages.system_messages_manager.get_active_messages()
            for message in active_system_messages:
                self.permanent_conversation.append_message(
                    SystemMessage(message.content))

    def add_system_message(
            self,
            system_message_content,
            is_clear_conversation_history=False):
        add_system_message_result = \
            self.conversation_and_system_messages.add_system_message(
                system_message_content,
                is_clear_conversation_history=is_clear_conversation_history)

        self.permanent_conversation.add_message(
            SystemMessage(system_message_content))

        return add_system_message_result

    def append_message(self, message: Message) -> None:
        self.conversation_and_system_messages.append_message(message)
        self.permanent_conversation.add_message(message)

        if message.role == "user":
            self._current_user_message = message
        elif message.role == "assistant":
            self._current_assistant_message = message
            if self._current_user_message is not None:
                self.permanent_conversation.append_message_pair(
                    self._current_user_message,
                    self._current_assistant_message)
                self._current_user_message = None
                self._current_assistant_message = None

    def append_general_message(self, message: Any) -> None:
        self.conversation_and_system_messages.append_general_message(message)

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
            self.conversation_and_system_messages.get_conversation_as_list_of_dicts()

    def get_permanent_conversation_messages(self):
        return self.permanent_conversation.messages

    def get_permanent_conversation_message_pairs(self):
        return self.permanent_conversation.message_pairs
