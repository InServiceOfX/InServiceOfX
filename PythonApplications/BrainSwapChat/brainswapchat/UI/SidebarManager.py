import streamlit as st
from commonapi.Messages import ConversationAndSystemMessages
from commonapi.FileIO import SystemMessagesFileIO

class SidebarManager:
    
    def render(self,
            conversation_and_system_messages: ConversationAndSystemMessages,
            configuration,
            application_paths=None):
        with st.sidebar:
            st.header("System Messages")

            self._render_active_system_messages(conversation_and_system_messages)
            st.divider()
            self._render_add_system_message(conversation_and_system_messages)
            st.divider()
            self._render_save_system_messages(conversation_and_system_messages, application_paths)
            st.divider()
            self._render_configuration_display(configuration)
    
    def _render_system_message_item(
            self,
            index: int,
            message,
            conversation_and_system_messages: ConversationAndSystemMessages):
        """Render individual system message item."""
        with st.expander(
            f"System Message {index+1} ({message.hash[:8]}...)", expanded=False):
            st.text_area(
                "Content",
                value=message.content,
                height=100,
                key=f"active_msg_{message.hash}",
                disabled=True
            )
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Deactivate", key=f"deactivate_{message.hash}"):
                    conversation_and_system_messages.system_messages_manager.toggle_message_to_inactive(
                        message.hash)
                    st.rerun()
            
            with col2:
                if st.button("Delete", key=f"delete_{message.hash}"):
                    conversation_and_system_messages.system_messages_manager.remove_message(
                        message.hash)
                    st.rerun()

    def _render_active_system_messages(
            self,
            conversation_and_system_messages: ConversationAndSystemMessages):
        active_messages = \
            conversation_and_system_messages.system_messages_manager.get_active_messages()
        
        if active_messages:
            st.subheader("Active System Messages:")
            for i, msg in enumerate(active_messages):
                self._render_system_message_item(
                    i,
                    msg,
                    conversation_and_system_messages)
        else:
            st.info("No active system messages")

    def _render_add_system_message(
            self,
            conversation_and_system_messages: ConversationAndSystemMessages):
        st.subheader("Add New System Message")
        
        new_system_message = st.text_area(
            "System Message Content",
            placeholder="Enter a new system message...",
            height=150,
            key="new_system_message"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Add & Activate", type="primary"):
                if new_system_message.strip():
                    result = conversation_and_system_messages.add_system_message(
                        new_system_message.strip(),
                        is_clear_conversation_history=False)

                    if result:
                        st.success("System message added and activated!")
                        # Use st.rerun() to clear the input instead of modifying
                        # session state
                        st.rerun()
                    else:
                        st.error("Message already exists!")
                else:
                    st.error("Please enter a system message")                
        
        with col2:
            if st.button("Reset Conversation"):
                conversation_and_system_messages.clear_conversation_history(
                    is_keep_active_system_messages=True)
                st.success("Conversation reset with active system messages!")
                st.rerun()

    def _render_save_system_messages(
            self,
            conversation_and_system_messages: ConversationAndSystemMessages,
            application_paths):
        st.subheader("Save System Messages")
        
        if application_paths is None:
            st.warning(
                "Application paths not available. Cannot save system messages.")
            return

        # Get all recorded system messages (both active and inactive)
        all_messages = \
            conversation_and_system_messages.system_messages_manager.messages

        if not all_messages:
            st.info("No system messages to save")
            return

        save_button_key = "save_system_messages_button"
        save_status_flag = "system_messages_saved"

        # Check if save was successful in previous run
        if save_status_flag in st.session_state and \
            st.session_state[save_status_flag]:
            st.success("✅ System messages saved successfully!")
            if st.button("Reset Save Status"):
                st.session_state[save_status_flag] = False
                st.rerun()
        else:
            if st.button(
                "💾 Save System Messages",
                type="primary",
                key=save_button_key):
                try:
                    # Ensure the file path exists
                    application_paths.create_missing_files(
                        "system_messages_file_path")

                    file_io = SystemMessagesFileIO(
                        application_paths.system_messages_file_path)

                    success = file_io.save_messages(all_messages)

                    if success:
                        st.session_state[save_status_flag] = True
                        st.rerun()
                    else:
                        st.error(
                            "Failed to save system messages. Check file "
                            "permissions.")
                        print("Save failed")

                except Exception as e:
                    st.error(f"Error saving system messages: {str(e)}")

    def _render_configuration_display(self, configuration):
        st.subheader("Configuration")
        st.write(f"**Model:** {configuration.model}")
        st.write(f"**Temperature:** {configuration.temperature}")
        if hasattr(configuration, 'max_tokens') and configuration.max_tokens:
            st.write(f"**Max Tokens:** {configuration.max_tokens}")