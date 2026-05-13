"""Run from the application directory:

    cd /InServiceOfX/PythonApplications/CLIPDFColQwenIndexer
    python Executables/main_CLIPDFColQwenIndexer.py --currentpath

The directory must contain:
    Configurations/colqwen2_5_configuration.yml
    Configurations/pdf_index_configuration.yml
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


from clipdfcolqwenindexer import ApplicationPaths


def main_CLIPDFColQwenIndexer():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Use development configuration (look in the app dir).",
    )
    parser.add_argument(
        "--currentpath",
        action="store_true",
        help="Use the current working directory as the configuration base.",
    )
    parser.add_argument(
        "--configpath",
        type=str,
        nargs=1,
        metavar="PATH",
        help="Explicit configuration base path.",
    )

    args = parser.parse_args()
    configpath = args.configpath[0] if args.configpath else None

    application_paths = ApplicationPaths.create(
        is_development=args.dev,
        is_current_path=args.currentpath,
        configpath=configpath,
    )
    application_paths.add_libraries_to_path()

    from clipdfcolqwenindexer.CLIPDFColQwenIndexer import (
        CLIPDFColQwenIndexer,
    )

    cli = CLIPDFColQwenIndexer(application_paths)
    cli.run()


if __name__ == "__main__":
    main_CLIPDFColQwenIndexer()
