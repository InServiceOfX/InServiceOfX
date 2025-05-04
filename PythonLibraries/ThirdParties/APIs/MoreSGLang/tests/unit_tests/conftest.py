from pathlib import Path
import sys

# To obtain modules from moresglang
if (Path(__file__).resolve().parents[2]).exists():
	sys.path.append(
		str(Path(__file__).resolve().parents[2]))


# To obtain modules from CoreCode
if (Path(__file__).resolve().parents[5] / "CoreCode").exists():
	sys.path.append(
		str(Path(__file__).resolve().parents[5] / "CoreCode"))
else:
	path = Path(__file__).resolve().parents[5]
	error_message = (
		"CoreCode directory not found. Please ensure it exists at the expected "
		"location. Expected path directory: " + str(path)
	)
	raise Exception(error_message)
