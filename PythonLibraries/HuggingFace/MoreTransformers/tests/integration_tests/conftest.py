from pathlib import Path
import sys

# To obtain modules from MoreTransformers
if Path(__file__).resolve().parents[2].exists():
	sys.path.append(str(Path(__file__).resolve().parents[2]))

# To obtain modules from CoreCode
if (Path(__file__).resolve().parents[4] / "CoreCode").exists():
	sys.path.append(
		str(Path(__file__).resolve().parents[4] / "CoreCode"))

common_api_path = (
	Path(__file__).resolve().parents[4] / \
		"ThirdParties" / \
		"APIs" / \
		"CommonAPI")

if common_api_path.exists() and str(common_api_path) not in sys.path:
	sys.path.append(str(common_api_path))
