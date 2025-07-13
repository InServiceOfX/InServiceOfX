from pathlib import Path
import sys

# To obtain modules from CommonAPI
if Path(__file__).resolve().parents[1].exists():
	sys.path.append(str(Path(__file__).resolve().parents[1]))

# To obtain modules from CoreCode
if (Path(__file__).resolve().parents[4] / "CoreCode").exists():
	sys.path.append(
		str(Path(__file__).resolve().parents[4] / "CoreCode"))

common_api_tests_path = Path(__file__).resolve().parents[1] / "tests"

if common_api_tests_path.exists() and \
	str(common_api_tests_path) not in sys.path:
	sys.path.append(str(common_api_tests_path))