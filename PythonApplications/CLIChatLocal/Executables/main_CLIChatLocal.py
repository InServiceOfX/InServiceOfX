"""
Usage: This is one way it can work:
python main_CLIChatLocal.py
or
python main_CLIChatLocal.py --dev
where you run it from the CLIChatLocal/Executables subdirectory where this file
is stored.
"""
from pathlib import Path
from warnings import warn
import argparse
import asyncio
import sys

application_path = Path(__file__).resolve().parents[1]

if not application_path.exists():
    warn(f"Application path {application_path} does not exist")
elif not str(application_path) in sys.path:
    sys.path.append(str(application_path))

from clichatlocal import ApplicationPaths

async def main_CLIChatLocal_async(clichat_local):
    await clichat_local.setup_postgresql_resource_and_embedding()
    clichat_local.setup_PermanentConversation_RAG_tools()
    await clichat_local.run_async()

def main_CLIChatLocal():
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

    args = parser.parse_args()

    # Extract the configpath value
    configpath = args.configpath[0] if args.configpath else None

    application_paths = ApplicationPaths.create(
        is_development=args.dev,
        is_current_path=args.currentpath,
        configpath=configpath)

    application_paths.add_libraries_to_path()

    from clichatlocal.CLIChatLocal import CLIChatLocal

    cli_chat_local = CLIChatLocal(application_paths)

    asyncio.run(main_CLIChatLocal_async(cli_chat_local))

if __name__ == "__main__":

    main_CLIChatLocal()