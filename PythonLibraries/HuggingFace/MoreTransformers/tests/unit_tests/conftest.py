from pathlib import Path
import sys

# To obtain modules from MoreTransformers
if Path(__file__).resolve().parents[2].exists():
	sys.path.append(str(Path(__file__).resolve().parents[2]))