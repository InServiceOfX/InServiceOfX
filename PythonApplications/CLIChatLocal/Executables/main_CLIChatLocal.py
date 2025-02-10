"""
Usage: This is one way it can work:
python main_CLIChatLocal.py
or
python main_CLIChatLocal.py --dev
where you run it from the CLIChatLocal/Executables subdirectory wwhere this file
is stored.
"""
from pathlib import Path
import sys

application_path = Path(__file__).resolve().parents[1]

if not str(application_path) in sys.path:
    sys.path.append(str(application_path))

from clichatlocal.FileIO import ApplicationPaths

def main_CLIChatLocal():
    print("Hello World")

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dev', action='store_true',
        help='Use development configuration')
    args = parser.parse_args()

    application_paths = ApplicationPaths.create(is_development=args.dev)

    if not str(application_paths.third_party_paths["moresglang"]) \
        in sys.path:
        sys.path.append(
            str(application_paths.third_party_paths["moresglang"]))

if __name__ == "__main__":

    main_CLIChatLocal()