from pathlib import Path
import sys

# To obtain modules from Tools
tools_path = Path(__file__).resolve().parents[1]
if tools_path.exists():
    if str(tools_path) not in sys.path:
        sys.path.append(str(tools_path))
else:
    error_message = (
        "Tools directory not found. Please ensure it exists at the expected "
        "location. Expected path directory: " + str(tools_path)
	)
    raise Exception(error_message)