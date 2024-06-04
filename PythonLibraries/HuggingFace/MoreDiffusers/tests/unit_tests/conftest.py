from pathlib import Path
import sys

# To obtain modules from MoreDiffusers
if Path(__file__).resolve().parents[3].exists():
    sys.path.append(str(Path(__file__).resolve().parents[3]))

# To obtain modules from CoreCode
if (Path(__file__).resolve().parents[4] / "CoreCode").exists():
    sys.path.append(
        str(Path(__file__).resolve().parents[4] / "CoreCode"))
