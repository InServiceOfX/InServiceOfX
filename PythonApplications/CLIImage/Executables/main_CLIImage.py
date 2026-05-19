"""
Usage: This is one way it can work:
python main_CLIImage.py
or
python main_CLIImage.py --dev
where you run it from the CLIImage/Executables subdirectory where this file is
stored.
"""
from pathlib import Path
import argparse
import sys
from warnings import warn

application_path = Path(__file__).resolve().parents[1]

if not application_path.exists():
    warn(f"Application path {application_path} does not exist")
elif not str(application_path) in sys.path:
    sys.path.append(str(application_path))

from cliimage import ApplicationPaths

def main_CLIImage():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dev', action='store_true',
        help='Use development configuration')
    parser.add_argument(
        '--currentpath', action='store_true',
        help=(
            "Use current working directory for where configuration files are "
            "saved; overrides --dev"))
    parser.add_argument(
        '--configpath',
        type=str,
        nargs=1,
        metavar='PATH',
        help=(
            "Specify custom base configuration path (takes first argument if "
            "multiple provided)"))
    parser.add_argument(
        '--command',
        action='append',
        default=[],
        metavar='COMMAND',
        help=(
            "Run a CLIImage dot command non-interactively. Can be passed "
            "multiple times. Example: --command '.list_loras'"))

    args = parser.parse_args()

    # Extract the configpath value
    configpath = args.configpath[0] if args.configpath else None

    application_paths = ApplicationPaths.create(
        is_development=args.dev,
        is_current_path=args.currentpath,
        configpath=configpath)

    application_paths.add_libraries_to_path()

    from cliimage.CLIImage import CLIImage

    cli_image = CLIImage(application_paths)

    if args.command:
        return 0 if cli_image.run_commands(args.command) else 1

    cli_image.run()
    return 0

if __name__ == "__main__":

    raise SystemExit(main_CLIImage())
