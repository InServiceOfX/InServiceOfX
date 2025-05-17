from pathlib import Path
import sys

# To obtain modules from MoreInsightFace
if Path(__file__).resolve().parent.parent.parent.exists():
	sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

# To obtain modules from CoreCode
if (Path(__file__).resolve().parents[4] / "CoreCode").exists():
	sys.path.append(
		str(Path(__file__).resolve().parents[4] / "CoreCode"))

# To obtain modules from MoreComputervision
if (Path(__file__).resolve().parents[4] / "MoreComputerVision").exists():
	sys.path.append(
		str(Path(__file__).resolve().parents[4] / "MoreComputerVision"))
