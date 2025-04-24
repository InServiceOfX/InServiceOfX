"""
Usage: This is one way it can work:
python main_CLIChatLocal.py
or
python main_CLIChatLocal.py --dev
where you run it from the CLIChatLocal/Executables subdirectory where this file
is stored.
"""
from pathlib import Path
import argparse
import sys

application_path = Path(__file__).resolve().parents[1]

if not str(application_path) in sys.path:
    sys.path.append(str(application_path))

from clichatlocal import ApplicationPaths

def main_CLIChatLocal():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dev', action='store_true',
        help='Use development configuration')
    args = parser.parse_args()

    application_paths = ApplicationPaths.create(is_development=args.dev)

    if not str(application_paths.inhouse_library_paths["moresglang"]) \
        in sys.path:
        sys.path.append(
            str(application_paths.inhouse_library_paths["moresglang"]))

    if not str(application_paths.inhouse_library_paths["CommonAPI"]) \
        in sys.path:
        sys.path.append(
            str(application_paths.inhouse_library_paths["CommonAPI"]))

    if not str(application_paths.inhouse_library_paths["MoreTransformers"]) \
        in sys.path:
        sys.path.append(
            str(application_paths.inhouse_library_paths["MoreTransformers"]))

    if not str(application_paths.inhouse_library_paths["CoreCode"]) \
        in sys.path:
        sys.path.append(
            str(application_paths.inhouse_library_paths["CoreCode"]))

    from clichatlocal.CLIChatLocal import CLIChatLocal

    cli_chat_local = CLIChatLocal(
        application_paths.configuration_file_paths["llama3_configuration"],
        application_paths.configuration_file_paths[
            "llama3_generation_configuration"],
        application_paths.system_messages_file_path,
        application_paths.conversations_file_path)
    cli_chat_local.run()

if __name__ == "__main__":

    main_CLIChatLocal()