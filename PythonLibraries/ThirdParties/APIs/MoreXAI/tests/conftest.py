from pathlib import Path
import sys

# To obtain modules from MoreXAI
more_xai_src_path = Path(__file__).resolve().parents[1] / "src"
if more_xai_src_path.exists():
    if str(more_xai_src_path) not in sys.path:
        sys.path.append(str(more_xai_src_path))
else:
    raise FileNotFoundError(f"MoreXAI src path not found: {more_xai_src_path}")

# To obtain modules from CoreCode
corecode_path = Path(__file__).resolve().parents[4] / "CoreCode"
if (corecode_path).exists():
    if str(corecode_path) not in sys.path:
        sys.path.append(str(corecode_path))
else:
    raise FileNotFoundError(f"CoreCode path not found: {corecode_path}")

commonapi_path = Path(__file__).resolve().parents[2] / "CommonAPI"
if (commonapi_path).exists():
    if str(commonapi_path) not in sys.path:
        sys.path.append(str(commonapi_path))
else:
    raise FileNotFoundError(f"CommonAPI path not found: {commonapi_path}")