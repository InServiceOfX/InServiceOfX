from pathlib import Path
import sys

# To obtain modules from MoreGroq
if Path(__file__).resolve().parents[2].exists():
	sys.path.append(str(Path(__file__).resolve().parents[2]))

# To obtain modules from CoreCode
if (Path(__file__).resolve().parents[5] / "CoreCode").exists():
	sys.path.append(
		str(Path(__file__).resolve().parents[5] / "CoreCode"))

# To obtain modules from CommonAPI
if (Path(__file__).resolve().parents[5] / \
	"ThirdParties" / \
	"APIs" / \
	"CommonAPI").exists():
	sys.path.append(
		str(Path(__file__).resolve().parents[5] / \
			"ThirdParties" / \
			"APIs" / \
			"CommonAPI"))
else:
	path = Path(__file__).resolve().parents[5]
	error_message = (
		"CommonAPI directory not found. Please ensure it exists at the expected"
		" location. Expected parent path directory: " + str(path)
	)
	raise Exception(error_message)
