from pathlib import Path
import pytest
import sys

# To obtain modules from MoreGroq
if Path(__file__).resolve().parents[2].exists():
	sys.path.append(str(Path(__file__).resolve().parents[2]))
