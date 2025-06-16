from pathlib import Path
import sys

# To obtain modules from CoreCode
core_code_path = Path(__file__).resolve().parents[2] / "CoreCode"
if core_code_path.exists():
    if str(core_code_path) not in sys.path:
        sys.path.append(str(core_code_path))
else:
    error_message = (
        "CoreCode directory not found. Please ensure it exists at the expected "
        "location. Expected path directory: " + str(core_code_path)
	)
    raise Exception(error_message)