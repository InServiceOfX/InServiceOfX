"""
USAGE:

in PythonApplications/BrainSwapCaht/
streamlit run brainswapchat/local_streamlit_app.py
"""

from pathlib import Path
import sys

BrainSwapChat_path = Path(__file__).resolve().parents[1]

if str(BrainSwapChat_path) not in sys.path:
    sys.path.append(str(BrainSwapChat_path))

from brainswapchat import SetupInternalModules
SetupInternalModules()

from brainswapchat.BrainSwapChatApp import BrainSwapChatApp

from corecode.Utilities import (get_environment_variable, load_environment_file)

from moregroq.Configuration import GroqClientConfiguration
from moregroq.Wrappers import GroqAPIWrapper

def local_streamlit_app_main():
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

    brain_swap_chat_app = BrainSwapChatApp(
        groq_client_configuration,
        groq_api_wrapper)
    brain_swap_chat_app.run()

if __name__ == "__main__":
    local_streamlit_app_main()
