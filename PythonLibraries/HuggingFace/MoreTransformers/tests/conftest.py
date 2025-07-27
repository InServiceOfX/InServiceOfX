from pathlib import Path
import sys

corecode_path = Path(__file__).resolve().parents[3] / "CoreCode"

if corecode_path.exists() and corecode_path not in sys.path:
	sys.path.append(str(corecode_path))
elif not corecode_path.exists():
	raise FileNotFoundError(f"CoreCode path {corecode_path} not found")

commonapi_path = \
	Path(__file__).resolve().parents[3] / "ThirdParties" / "APIs" / "CommonAPI"

if commonapi_path.exists() and commonapi_path not in sys.path:
	sys.path.append(str(commonapi_path))
elif not commonapi_path.exists():
	raise FileNotFoundError(f"commonapi path {commonapi_path} not found")

# # To obtain modules from MoreTransformers
# if Path(__file__).resolve().parents[2].exists():
# 	sys.path.append(str(Path(__file__).resolve().parents[2]))
