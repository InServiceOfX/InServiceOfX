"""
Usage: This is one way it can work:
python main_CLIVideo.py
or
python main_CLIVideo.py --dev
where you run it from the CLIVideo/Executables subdirectory wwhere this file
is stored.
"""
from pathlib import Path
import sys

application_path = Path(__file__).resolve().parents[1]

if not str(application_path) in sys.path:
    sys.path.append(str(application_path))
    from clivideo.Utilities import Printing
    Printing.print_info(f"Added {application_path} to sys.path")

from clivideo.FileIO import ApplicationPaths
from clivideo.Utilities import load_environment_file
from clivideo.Utilities import Printing

def main_CLIVideo():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dev', action='store_true',
        help='Use development configuration')
    args = parser.parse_args()

    application_paths = ApplicationPaths.create(is_development=args.dev)

    if not application_paths.environment_file_path.exists():
        Printing.print_info(
            f"\nEnvironment file not found at "
            f"<ansiyellow>{application_paths.environment_file_path}</ansiyellow>")
    else:
        Printing.print_info(
            f"Loading environment variables from "
            f"<ansiyellow>{application_paths.environment_file_path}</ansiyellow>")

    load_environment_file(application_paths.environment_file_path)

    if not str(application_paths.third_party_paths["morelumaai"]) in sys.path:
        Printing.print_info(
            f"Adding <ansicyan>{application_paths.third_party_paths['morelumaai']}</ansicyan> "
            "to sys.path")
        sys.path.append(str(application_paths.third_party_paths["morelumaai"]))

    if not str(application_paths.third_party_paths["morefal"]) in sys.path:
        Printing.print_info(
            f"Adding <ansicyan>{application_paths.third_party_paths['morefal']}</ansicyan> "
            "to sys.path")
        sys.path.append(str(application_paths.third_party_paths["morefal"]))

    # Because CLIVideo depends on morelumaai, we need to import here.
    from clivideo.CLIVideo import CLIVideo

    cli_video = CLIVideo(
        application_paths.configuration_file_path,
        application_paths.lumaai_configuration_file_path)
    cli_video.run()


if __name__ == "__main__":
    main_CLIVideo()