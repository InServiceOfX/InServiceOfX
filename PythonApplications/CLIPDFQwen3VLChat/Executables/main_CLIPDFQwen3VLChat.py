"""Run from the application directory:

    cd /InServiceOfX/PythonApplications/CLIPDFQwen3VLChat
    python Executables/main_CLIPDFQwen3VLChat.py --currentpath

Or specify an explicit configuration directory:

    python Executables/main_CLIPDFQwen3VLChat.py --configpath /some/dir

The directory must contain:
    Configurations/qwen3vl_configuration.yml
    Configurations/pdf_chat_configuration.yml
"""
from pathlib import Path
from warnings import warn
import argparse
import sys


application_path = Path(__file__).resolve().parents[1]

if not application_path.exists():
    warn(f"Application path {application_path} does not exist")
elif str(application_path) not in sys.path:
    sys.path.append(str(application_path))


from clipdfqwen3vlchat import ApplicationPaths


def main_CLIPDFQwen3VLChat():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Use development configuration (look in the app dir).",
    )
    parser.add_argument(
        "--currentpath",
        action="store_true",
        help=(
            "Use the current working directory as the configuration base; "
            "overrides --dev."
        ),
    )
    parser.add_argument(
        "--configpath",
        type=str,
        nargs=1,
        metavar="PATH",
        help=(
            "Explicit configuration base path; takes precedence over --dev "
            "and --currentpath."
        ),
    )

    args = parser.parse_args()
    configpath = args.configpath[0] if args.configpath else None

    application_paths = ApplicationPaths.create(
        is_development=args.dev,
        is_current_path=args.currentpath,
        configpath=configpath,
    )

    application_paths.add_libraries_to_path()

    from clipdfqwen3vlchat.CLIPDFQwen3VLChat import CLIPDFQwen3VLChat

    cli = CLIPDFQwen3VLChat(application_paths)
    cli.run()


if __name__ == "__main__":
    main_CLIPDFQwen3VLChat()
