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

commonapi_path = tools_path.parents[0] / "ThirdParties" / "APIs" / "CommonAPI"
if commonapi_path.exists():
    if str(commonapi_path) not in sys.path:
        sys.path.append(str(commonapi_path))
else:
    print(
        f"CommonAPI directory not found. Please ensure it exists at the "
        "expected location. Expected path directory: " + str(commonapi_path))

corecode_path = tools_path.parents[0] / "CoreCode"
if corecode_path.exists():
    if str(corecode_path) not in sys.path:
        sys.path.append(str(corecode_path))
else:
    print(
        f"CoreCode directory not found. Please ensure it exists at the "
        "expected location. Expected path directory: " + str(corecode_path))