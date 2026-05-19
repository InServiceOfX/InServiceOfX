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
    parser.add_argument('--prompt', type=str, help="Override prompt for this run")
    parser.add_argument(
        '--prompt-2',
        type=str,
        help="Override second prompt for this run")
    parser.add_argument(
        '--negative-prompt',
        type=str,
        help="Override negative prompt for this run")
    parser.add_argument(
        '--negative-prompt-2',
        type=str,
        help="Override second negative prompt for this run")
    parser.add_argument(
        '--output-path',
        type=str,
        help="Override temporary_save_path for this run")
    parser.add_argument('--height', type=int, help="Override image height")
    parser.add_argument('--width', type=int, help="Override image width")
    parser.add_argument(
        '--steps',
        type=int,
        help="Override num_inference_steps for this run")
    parser.add_argument(
        '--guidance-scale',
        type=float,
        help="Override guidance_scale for this run")
    parser.add_argument(
        '--true-cfg-scale',
        type=float,
        help="Override true_cfg_scale for this run")
    parser.add_argument('--seed', type=int, help="Override seed for this run")
    parser.add_argument(
        '--batch-images',
        type=int,
        help="Override batch number_of_images for this run")

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
    cli_image.apply_overrides(args)

    if args.command:
        return 0 if cli_image.run_commands(args.command) else 1

    cli_image.run()
    return 0

if __name__ == "__main__":

    raise SystemExit(main_CLIImage())
