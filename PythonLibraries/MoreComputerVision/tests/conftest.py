from pathlib import Path
import sys

# To obtain modules from MoreComputerVision
if Path(__file__).resolve().parents[2].exists():
	sys.path.append(str(Path(__file__).resolve().parents[3]))
