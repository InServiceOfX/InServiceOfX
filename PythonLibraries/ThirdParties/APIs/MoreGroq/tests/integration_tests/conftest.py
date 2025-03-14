from pathlib import Path
import pytest
import sys

# To obtain modules from MoreGroq
if Path(__file__).resolve().parents[2].exists():
	sys.path.append(str(Path(__file__).resolve().parents[2]))

# To obtain modules from CoreCode
if (Path(__file__).resolve().parents[5] / "CoreCode").exists():
	sys.path.append(
		str(Path(__file__).resolve().parents[5] / "CoreCode"))

# To obtain modules from CommonAPI
if (Path(__file__).resolve().parents[3] / "CommonAPI").exists():
	sys.path.append(
		str(Path(__file__).resolve().parents[3] / "CommonAPI"))

# To obtain TestUtilities
sys.path.append(str(Path(__file__).resolve().parent))
