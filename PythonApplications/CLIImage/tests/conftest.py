from pathlib import Path
import sys

cli_image_path = Path(__file__).resolve().parents[1]

# To obtain modules from CoreCode
if cli_image_path.exists():
    if str(cli_image_path) not in sys.path:
        sys.path.append(str(cli_image_path))
else:
    raise FileNotFoundError(
        f"CLIImage path {cli_image_path} not found")

corecode_path = cli_image_path.parents[1] / "PythonLibraries" / "CoreCode"

if corecode_path.exists():
    if str(corecode_path) not in sys.path:
        sys.path.append(str(corecode_path))
else:
    raise FileNotFoundError(
        f"CoreCode path {corecode_path} not found")

morediffusers_path = cli_image_path.parents[1] / "PythonLibraries" / "HuggingFace" / "MoreDiffusers"

if morediffusers_path.exists():
    if str(morediffusers_path) not in sys.path:
        sys.path.append(str(morediffusers_path))
else:
    raise FileNotFoundError(
        f"MoreDiffusers path {morediffusers_path} not found")

moretransformers_path = cli_image_path.parents[1] / "PythonLibraries" / "HuggingFace" / "MoreTransformers"