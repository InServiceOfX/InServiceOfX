import streamlit as st
from brainswapchat.UI import SidebarManager, ChatInterface, GroqModelSelector
from commonapi.Messages import ConversationAndSystemMessages

class BrainSwapChatApp:
    def __init__(self, groq_client_configuration, groq_service, model_selector):
        self.sidebar_manager = SidebarManager()
        self.chat_interface = ChatInterface()
        self.groq_model_selector = GroqModelSelector()

        if "conversation_and_system_messages" not in st.session_state:
            st.session_state.conversation_and_system_messages = \
                ConversationAndSystemMessages()
            st.session_state.conversation_and_system_messages.add_default_system_message()

        if "groq_client_configuration" not in st.session_state:
            st.session_state.groq_client_configuration = groq_client_configuration

        if "groq_api_wrapper" not in st.session_state:
            st.session_state.groq_api_wrapper = groq_service

        if "model_selector" not in st.session_state:
            st.session_state.model_selector = model_selector

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
            st.session_state.groq_client_configuration,
            application_paths)
        self.chat_interface.render(
            st.session_state.conversation_and_system_messages,
            st.session_state.groq_api_wrapper)

        self.groq_model_selector.render_in_topbar(
            st.session_state.groq_client_configuration,
            st.session_state.groq_api_wrapper,
            st.session_state.model_selector
        )