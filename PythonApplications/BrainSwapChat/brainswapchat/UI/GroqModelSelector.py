import streamlit as st
from typing import Optional, Dict, Any
from moregroq.Configuration import GroqClientConfiguration

class GroqModelSelector:
    """Handles Groq model selection for BrainSwapChat."""
    
    @staticmethod
    def _render_model_selector(model_selector) \
        -> Optional[str]:
        """Render the model selection dropdown and return selected model if
        changed."""

        print("render_model_selector is called")

        models_list = model_selector.get_all_available_models()

        # Create dropdown options with display names
        options = []
        model_ids = []
        current_index = 0
        
        for i, model in enumerate(models_list):
            display_name = \
                f"{model['id']} (context window: {model['context_window']})"
            options.append(display_name)
            model_ids.append(model['id'])

            if model['id'] == st.session_state.model_selector.current_model:
                print("found current model", model['id'], i)
                current_index = i

        print(f"current_index: {current_index}")
        print(f"current_model: {st.session_state.model_selector.current_model}")

        # Use st.selectbox with key and index to let Streamlit manage the state
        selected_option = st.selectbox(
            "Select Model:",
            options=options,
            index=current_index,
            key="groq_model_selector"
        )

        print("You selected: ", selected_option)

        selected_model_id = model_ids[options.index(selected_option)]

        print(f"selected_model_id: {selected_model_id}")

        # Check if the selection changed
        if selected_model_id != st.session_state.model_selector.current_model:
            st.session_state.model_selector.current_model = selected_model_id
            return selected_model_id
        else:
            return None

    @staticmethod
    def render_in_topbar(
            groq_client_configuration,
            groq_api_wrapper,
            model_selector) -> bool:
        """Render model selector in topbar and handle model switching."""
        # Create a container for the model selector next to the sidebar

        print("render_in_topbar is called")

        with st.container():
            # Adjust ratio as needed
            col1, col2 = st.columns([1, 4])
            
            with col1:
                # Empty space to align with sidebar
                st.write("")
            
            with col2:
                # Render model selector and get selected model if changed
                selected_model = GroqModelSelector._render_model_selector(
                    model_selector=model_selector)
                
                # If model changed,
                if selected_model:
                    st.info(
                        f"🔄 Switching to {selected_model}...")
                    
                    # Get the context window for the selected model
                    context_window = \
                        model_selector.get_context_window_by_model_name(
                            selected_model)

                    new_configuration = \
                        GroqModelSelector._update_groq_client_configuration(
                            current_configuration=groq_client_configuration,
                            selected_model=selected_model,
                            selected_model_max_tokens=context_window)

                    new_configuration.update_chat_completion_configuration(
                        groq_api_wrapper.configuration)

                    # Update the passed references
                    groq_client_configuration = new_configuration
                                        
                    st.success(
                        f"✅ Model switched to {selected_model} (max_tokens: {context_window})")
                    st.rerun()
                    return True
        
        return False

    @staticmethod
    def _update_groq_client_configuration(
            current_configuration: GroqClientConfiguration,
            selected_model: str,
            selected_model_max_tokens: Optional[int] = None) \
                -> GroqClientConfiguration:

        if current_configuration is None:
            raise ValueError("current_configuration is None")
        # Determine max_tokens value
        max_tokens = None
        if selected_model_max_tokens is not None:
            max_tokens = selected_model_max_tokens
        else:
            max_tokens = current_configuration.max_tokens
        
        new_config = GroqClientConfiguration(
            model=selected_model,
            temperature=current_configuration.temperature,
            max_tokens=max_tokens
        )
        
        return new_config