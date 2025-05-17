from pathlib import Path
import sys

# To obtain modules from MoreInsightFace
if Path(__file__).resolve().parent.parent.parent.exists():
	sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

# To obtain modules from CoreCode
if (Path(__file__).resolve()
    .parent.parent.parent.parent.parent / "CoreCode").exists():
	sys.path.append(
		str(Path(__file__).resolve()
            .parent.parent.parent.parent.parent / "CoreCode"))

# To obtain modules from HuggingFace/moreDiffusers
if (Path(__file__).resolve()
    .parents[4] / "HuggingFace" / "MoreDiffusers").exists():
    sys.path.append(
        str(Path(__file__).resolve()
            .parents[4] / "HuggingFace" / "MoreDiffusers"))

# To obtain modules from InstantID

if (Path(__file__).resolve().parents[6] / "ThirdParty" / "InstantID").exists():
    sys.path.append(
        str(Path(__file__).resolve().parents[6] / "ThirdParty" / "InstantID"))
