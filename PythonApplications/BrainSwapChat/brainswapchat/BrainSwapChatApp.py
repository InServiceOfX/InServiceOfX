import streamlit as st
from brainswapchat.UI import SidebarManager, ChatInterface
from commonapi.Messages import ConversationAndSystemMessages

class BrainSwapChatApp:
    def __init__(self, configuration, groq_service):
        self.sidebar_manager = SidebarManager()
        self.chat_interface = ChatInterface()
        self._configuration = configuration
        self._groq_service = groq_service
        if "conversation_and_system_messages" not in st.session_state:
            st.session_state.conversation_and_system_messages = \
                ConversationAndSystemMessages()
            st.session_state.conversation_and_system_messages.add_default_system_message()

    def _render_header(self):
        """Render application header."""

        st.title("💬 Chatbot")
        st.write(
            (
                "This is a simple chatbot that uses Groq's models to generate "
                "responses and swap system messages on the fly. "
                "To use this app, you need to provide a Groq API key, which "
                "you can get [here](https://console.groq.com/keys). "
            )
        )

    def run(self, application_paths=None):
        self._render_header()

        self.sidebar_manager.render(
            st.session_state.conversation_and_system_messages,
            self._configuration,
            application_paths)
        self.chat_interface.render(
            st.session_state.conversation_and_system_messages,
            self._groq_service)