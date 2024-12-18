from pathlib import Path
import sys

project_path = Path(__file__).resolve().parents[3]
corecode_directory = project_path / "PythonLibraries" / "CoreCode"
clichat_directory = project_path / "PythonApplications" / "CLIChat"

if not str(corecode_directory) in sys.path:
    sys.path.append(str(corecode_directory))

if not str(clichat_directory) in sys.path:
    sys.path.append(str(clichat_directory))

from clichat.Configuration import Configuration
from clichat.Chatbot import Chatbot

def main_Chatbot():
    configuration = Configuration()
    chatbot = Chatbot(configuration=configuration)
    chatbot.run()

if __name__ == "__main__":

    main_Chatbot()