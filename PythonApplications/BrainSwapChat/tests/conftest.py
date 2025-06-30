from pathlib import Path
import sys

tests_path = Path(__file__).resolve().parent

if str(tests_path) not in sys.path:
    sys.path.append(str(tests_path))

project_path = Path(__file__).resolve().parents[1]

if str(project_path) not in sys.path:
    sys.path.append(str(project_path))

repo_path = Path(__file__).resolve().parents[3]

corecode_path = repo_path / "PythonLibraries" / "CoreCode"

if not corecode_path.exists():
    raise FileNotFoundError(f"CoreCode path not found: {corecode_path}")

if str(corecode_path) not in sys.path:
    sys.path.append(str(corecode_path))

commonapi_path = repo_path / "PythonLibraries" / "ThirdParties" / "APIs" / \
    "CommonAPI"

if str(commonapi_path) not in sys.path:
    sys.path.append(str(commonapi_path))

moregroq_path = repo_path / "PythonLibraries" / "ThirdParties" / "APIs" / \
    "MoreGroq"

if str(moregroq_path) not in sys.path:
    sys.path.append(str(moregroq_path))

