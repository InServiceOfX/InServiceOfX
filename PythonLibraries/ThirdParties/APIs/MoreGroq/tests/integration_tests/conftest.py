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

# To obtain modules from MoreTransformers
more_transformers_path = Path(__file__).resolve().parents[5] / \
	"HuggingFace" / "MoreTransformers"

if more_transformers_path.exists():
	sys.path.append(str(more_transformers_path))

# To obtain TestUtilities
sys.path.append(str(Path(__file__).resolve().parent))
