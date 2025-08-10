"""
USAGE:

in PythonApplications/BrainSwapChat/
streamlit run brainswapchat/local_streamlit_app.py
"""

from pathlib import Path
import sys

BrainSwapChat_path = Path(__file__).resolve().parents[1]

if str(BrainSwapChat_path) not in sys.path:
    sys.path.append(str(BrainSwapChat_path))

from brainswapchat import ApplicationPaths
from brainswapchat import SetupInternalModules
# This has to happen before the other imports.
SetupInternalModules()

from brainswapchat.BrainSwapChatApp import BrainSwapChatApp
from brainswapchat.SetupSystemMessages import setup_system_messages

from corecode.Utilities import (get_environment_variable, load_environment_file)

from moregroq.Applications import ModelSelector
from moregroq.Configuration import GroqClientConfiguration
from moregroq.Wrappers import GroqAPIWrapper

import streamlit as st

def local_streamlit_app_main():
    application_paths = ApplicationPaths.create_path_names(
        is_development=True)

    groq_client_configuration_path = BrainSwapChat_path / "Configurations" / \
        "groq_client_configuration.yml"

    if not groq_client_configuration_path.exists():
        raise FileNotFoundError(
            f"Groq client configuration file not found: {groq_client_configuration_path}")

    groq_client_configuration = GroqClientConfiguration.from_yaml(
        groq_client_configuration_path)

    load_environment_file()

    groq_api_wrapper = GroqAPIWrapper(
        api_key=get_environment_variable("GROQ_API_KEY"),
        groq_client_configuration=groq_client_configuration)

    model_selector = ModelSelector(
        api_key=get_environment_variable("GROQ_API_KEY"))

    model_selector.current_model = groq_client_configuration.model

    brain_swap_chat_app = BrainSwapChatApp(
        groq_client_configuration,
        groq_api_wrapper,
        model_selector)
    
    if "conversation_and_system_messages" not in st.session_state:
        raise RuntimeError(
            "Conversation and system messages not found in session state. "
            "BrainSwapChatApp __init__() should have created it.")

    setup_system_messages(
        application_paths,
        st.session_state.conversation_and_system_messages)

    brain_swap_chat_app.run(application_paths)

if __name__ == "__main__":
    local_streamlit_app_main()
