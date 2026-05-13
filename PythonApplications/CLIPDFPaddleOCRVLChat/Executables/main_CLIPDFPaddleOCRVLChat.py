"""Run from the application directory:

    cd /InServiceOfX/PythonApplications/CLIPDFPaddleOCRVLChat
    python Executables/main_CLIPDFPaddleOCRVLChat.py --currentpath

Requires a running PaddleOCR-VL vLLM server, usually:

    vllm serve /Data/Models/Multimodal/PaddlePaddle/PaddleOCR-VL-1.5 \
      --served-model-name paddleocr-vl --port 8080 --max-model-len 8192 \
      --gpu-memory-utilization 0.85 --enforce-eager
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


from clipdfpaddleocrvlchat import ApplicationPaths


def main_CLIPDFPaddleOCRVLChat():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", action="store_true")
    parser.add_argument("--currentpath", action="store_true")
    parser.add_argument("--configpath", type=str, nargs=1, metavar="PATH")

    args = parser.parse_args()
    configpath = args.configpath[0] if args.configpath else None

    application_paths = ApplicationPaths.create(
        is_development=args.dev,
        is_current_path=args.currentpath,
        configpath=configpath,
    )
    application_paths.add_libraries_to_path()

    from clipdfpaddleocrvlchat.CLIPDFPaddleOCRVLChat import (
        CLIPDFPaddleOCRVLChat,
    )

    cli = CLIPDFPaddleOCRVLChat(application_paths)
    cli.run()


if __name__ == "__main__":
    main_CLIPDFPaddleOCRVLChat()
