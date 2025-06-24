import streamlit as st
from commonapi.Messages import (
    ConversationAndSystemMessages,
    UserMessage,
    AssistantMessage)

class ChatInterface:
    """Manages the main chat interface."""
    
    def render(
            self,
            conversation: ConversationAndSystemMessages,
            groq_service):
        """Render the complete chat interface."""
        self._render_chat_messages(conversation)
        self._render_chat_input(conversation, groq_service)
    
    def _render_chat_messages(
            self,
            conversation: ConversationAndSystemMessages):
        """Render existing chat messages."""
        conversation_dicts = conversation.get_conversation_as_list_of_dicts()
        
        for message in conversation_dicts:
            # Skip system messages in the main chat display
            if message["role"] != "system":
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
    
    def _render_chat_input(
            self,
            conversation: ConversationAndSystemMessages,
            groq_service):
        """Render chat input and handle user interactions.

        Create a chat input field to allow the user to enter a message. This
        will display automatically at the bottom of the page. Streamlit's
        reactive model automatically reruns this when inputs change.
        """
        if prompt := st.chat_input("What is up?"):
            self._handle_user_input(prompt, conversation, groq_service)
    
    def _handle_user_input(
            self,
            prompt: str,
            conversation: ConversationAndSystemMessages,
            groq_service):
        """Handle user input and generate response."""
        # Add user message to conversation
        user_message = UserMessage(content=prompt)
        conversation.append_message(user_message)
        
        # Display the user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        self._generate_and_display_response(conversation, groq_service)
    
    def _generate_and_display_response(
            self,
            conversation: ConversationAndSystemMessages,
            groq_service):
        """Generate a response using the API call and display the response."""
        try:
            # Get conversation as list of dicts for API call
            messages_for_api = conversation.get_conversation_as_list_of_dicts()
            
            response = groq_service.create_chat_completion(messages_for_api)
            
            # Extract and display response
            if response and hasattr(response, 'choices') and \
                len(response.choices) > 0:
                assistant_message_content = response.choices[0].message.content
                
                # Create and add assistant message
                assistant_message = AssistantMessage(
                    content=assistant_message_content)
                conversation.append_message(assistant_message)
                
                # Display the response
                with st.chat_message("assistant"):
                    st.markdown(assistant_message_content)
            else:
                with st.chat_message("assistant"):
                    st.error("response: " + str(response))
                    st.error("No response received from API call")
                    
        except Exception as e:
            with st.chat_message("assistant"):
                st.error(f"Error: {str(e)}")