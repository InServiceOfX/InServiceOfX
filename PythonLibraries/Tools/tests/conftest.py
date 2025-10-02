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

tools_tests_path = tools_path / "tests"
if tools_tests_path.exists():
    if str(tools_tests_path) not in sys.path:
        sys.path.append(str(tools_tests_path))
else:
    print(
        f"Tools tests directory not found. Please ensure it exists at the "
        "expected location. Expected path directory: " + str(tools_tests_path))

commonapi_path = tools_path.parents[0] / "ThirdParties" / "APIs" / "CommonAPI"
if commonapi_path.exists():
    if str(commonapi_path) not in sys.path:
        sys.path.append(str(commonapi_path))
else:
    print(
        f"CommonAPI directory not found. Please ensure it exists at the "
        "expected location. Expected path directory: " + str(commonapi_path))

commonapi_test_data_path = commonapi_path / "tests" / "TestData"

if commonapi_test_data_path.exists():
    if str(commonapi_test_data_path) not in sys.path:
        sys.path.append(str(commonapi_test_data_path))
else:
    print(
        f"CommonAPI test data path does not exist. Please ensure it exists at "
        "the expected location. Expected path directory: " + str(commonapi_test_data_path))

corecode_path = tools_path.parents[0] / "CoreCode"
if corecode_path.exists():
    if str(corecode_path) not in sys.path:
        sys.path.append(str(corecode_path))
else:
    print(
        f"CoreCode directory not found. Please ensure it exists at the "
        "expected location. Expected path directory: " + str(corecode_path))

embeddings_path = tools_path.parents[0] / "Embeddings"
if embeddings_path.exists():
    if str(embeddings_path) not in sys.path:
        sys.path.append(str(embeddings_path))
else:
    print(
        f"Embeddings directory not found. Please ensure it exists at the "
        "expected location. Expected path directory: " + str(embeddings_path))
    
moretransformers_path = tools_path.parents[0] / "HuggingFace" / \
    "MoreTransformers"
if moretransformers_path.exists():
    if str(moretransformers_path) not in sys.path:
        sys.path.append(str(moretransformers_path))
else:
    print(
        f"MoreTransformers directory not found. Please ensure it exists at the "
        "expected location. Expected path directory: " + \
            str(moretransformers_path))