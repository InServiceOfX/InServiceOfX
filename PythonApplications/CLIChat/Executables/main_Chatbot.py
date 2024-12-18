from pathlib import Path
import sys

application_path = Path(__file__).resolve().parents[1]
project_path = Path(__file__).resolve().parents[3]
corecode_path = project_path / "PythonLibraries" / "CoreCode"

if not str(application_path) in sys.path:
    sys.path.append(str(application_path))

if not str(corecode_path) in sys.path:
    sys.path.append(str(corecode_path))

from clichat.Configuration import Configuration
from clichat.Chatbot import Chatbot

from corecode.Utilities import load_environment_file

def main_Chatbot():
    # Load the environment variables.
    environment_file_path = application_path / "Configurations" / ".env"
    if not environment_file_path.exists():
        print(f"\nEnvironment file not found at {environment_file_path}")
    else:
        print(f"Loading environment variables from {environment_file_path}")

    load_environment_file(environment_file_path)

    configuration = Configuration()
    print(
        "Found and loaded configuration at:",
        configuration.configuration_path)
    chatbot = Chatbot(configuration=configuration)
    chatbot.run()

if __name__ == "__main__":

    main_Chatbot()