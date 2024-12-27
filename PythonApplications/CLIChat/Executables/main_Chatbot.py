from pathlib import Path
import sys

application_path = Path(__file__).resolve().parents[1]
project_path = Path(__file__).resolve().parents[3]
corecode_path = project_path / "PythonLibraries" / "CoreCode"
more_groq_path = project_path / "PythonLibraries" / "ThirdParties" / "MoreGroq"

if not str(application_path) in sys.path:
    sys.path.append(str(application_path))

if not str(corecode_path) in sys.path:
    sys.path.append(str(corecode_path))

if not str(more_groq_path) in sys.path:
    sys.path.append(str(more_groq_path))

from clichat.Configuration import Configuration
from clichat.Chatbot import Chatbot
from clichat.Utilities import Printing

from corecode.Utilities import load_environment_file

def main_Chatbot():
    # Load the environment variables.
    environment_file_path = application_path / "Configurations" / ".env"
    if not environment_file_path.exists():
        Printing.print_info(
            f"\nEnvironment file not found at {environment_file_path}")
    else:
        Printing.print_info(
            f"Loading environment variables from {environment_file_path}")

    load_environment_file(environment_file_path)

    configuration = Configuration()
    Printing.print_info(
        f"Found and loaded configuration at: {configuration.configuration_path}")
    chatbot = Chatbot(configuration=configuration)
    chatbot.run()

if __name__ == "__main__":

    main_Chatbot()