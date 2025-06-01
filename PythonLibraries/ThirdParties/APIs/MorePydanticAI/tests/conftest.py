from pathlib import Path
import pytest
import sys

# To obtain modules from CoreCode
if (Path(__file__).resolve().parents[4] / "CoreCode").exists():
	sys.path.append(
		str(Path(__file__).resolve().parents[4] / "CoreCode"))

