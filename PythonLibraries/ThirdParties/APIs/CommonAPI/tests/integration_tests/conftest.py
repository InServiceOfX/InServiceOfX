from pathlib import Path
import sys

# To obtain modules from CommonAPI
if Path(__file__).resolve().parents[2].exists():
	sys.path.append(str(Path(__file__).resolve().parents[2]))

# To obtain modules from CoreCode
if (Path(__file__).resolve().parents[5] / "CoreCode").exists():
	sys.path.append(
		str(Path(__file__).resolve().parents[5] / "CoreCode"))
