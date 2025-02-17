"""
Usage: This is one way it can work:
python main_Chatbot.py
or
python main_Chatbot.py --dev
where you run it from the CLIChat/Executables subdirectory wwhere this file
is stored.
"""
from pathlib import Path
import sys

application_path = Path(__file__).resolve().parents[1]
project_path = Path(__file__).resolve().parents[3]
more_groq_path = project_path / "PythonLibraries" / "ThirdParties" / "APIs" / "MoreGroq"

if not str(application_path) in sys.path:
    sys.path.append(str(application_path))

if not str(more_groq_path) in sys.path:
    sys.path.append(str(more_groq_path))

from clichat.Configuration import Configuration
from clichat.Chatbot import Chatbot
from clichat.Utilities import (Printing, load_environment_file)

def main_Chatbot():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dev', action='store_true',
        help='Use development configuration')
    args = parser.parse_args()

    if args.dev:
        environment_file_path = application_path / "Configurations" / ".env"
        config_path = application_path / "Configurations" / "clichat_configuration.yml"
    else:
        config_dir = Path.home() / ".config" / "clichat"
        environment_file_path = config_dir / "Configurations" / ".env"
        config_path = config_dir / "Configurations" / "clichat_configuration.yml"

    if not environment_file_path.exists():
        Printing.print_info(
            f"\nEnvironment file not found at {environment_file_path}")
    else:
        Printing.print_info(
            f"Loading environment variables from {environment_file_path}")

    load_environment_file(environment_file_path)

    configuration = Configuration(config_path)
    Printing.print_info(
        f"Found and loaded configuration at: {configuration.configuration_path}")
    chatbot = Chatbot(configuration=configuration)
    chatbot.run()

if __name__ == "__main__":

    main_Chatbot()