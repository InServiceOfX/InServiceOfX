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
        
        print("render_in_topbar is called")

        # Create a compact, flat design with custom CSS
        st.markdown("""
        <style>
        .model-selector-flat {
            background: linear-gradient(90deg, #f8f9fa 0%, #e9ecef 100%);
            border: 1px solid #dee2e6;
            border-radius: 6px;
            padding: 8px 12px;
            margin: 8px 0;
        }
        .model-selector-flat .stSelectbox {
            margin: 0;
        }
        .small-text {
            font-size: 11px;
            color: #6c757d;
        }
        </style>
        """, unsafe_allow_html=True)

        # Create a single row layout
        with st.container():
            st.markdown('<div class="model-selector-flat">', unsafe_allow_html=True)
            
            # Single row: Model dropdown and current info
            col1, col2, col3 = st.columns([1, 3, 1])
            
            with col1:
                st.markdown("🤖 **Model:**", unsafe_allow_html=True)
            
            with col2:
                # Model dropdown
                selected_model = GroqModelSelector._render_model_selector(
                    model_selector=model_selector)
            
            with col3:
                # Current model info (compact)
                if st.session_state.model_selector.current_model:
                    current_model = st.session_state.model_selector.current_model
                    context_window = model_selector.get_context_window_by_model_name(current_model)
                    # Show first 8 characters of model name
                    model_display = current_model[:32] + "..." if len(current_model) > 8 else current_model
                    st.markdown(f"<div class='small-text'>{model_display}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='small-text'>Context: {context_window:,}</div>", unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # If model changed,
        if selected_model:
            # Use a more subtle notification
            with st.container():
                st.markdown(f"🔄 **Switching to {selected_model}**")
                
                context_window = \
                    model_selector.get_context_window_by_model_name(
                        selected_model)

                # TODO: Do more testing for when context window exceeded max
                # tokens allowed for each model.
                new_configuration = \
                    GroqModelSelector._update_groq_client_configuration(
                        current_configuration=groq_client_configuration,
                        selected_model=selected_model,
                        selected_model_max_tokens=None)

                new_configuration.update_chat_completion_configuration(
                    groq_api_wrapper.configuration)

                groq_client_configuration = new_configuration
                                
                st.success(f"✅ **{selected_model}** ready")
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