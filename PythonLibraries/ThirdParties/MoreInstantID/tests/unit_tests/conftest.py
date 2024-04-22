from pathlib import Path
import sys

# To obtain modules from MoreInstantID
if Path(__file__).resolve().parent.parent.parent.exists():
	sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

# To obtain modules from CoreCode
if (Path(__file__).resolve().parent.parent.parent.parent.parent / "CoreCode").exists():
	sys.path.append(
		str(Path(__file__).resolve().parent.parent.parent.parent.parent / "CoreCode"))
