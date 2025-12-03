from pathlib import Path
import sys

# To obtain modules from CommonAPI
if Path(__file__).resolve().parents[1].exists():
	sys.path.append(str(Path(__file__).resolve().parents[1]))

# To obtain modules from CoreCode
if (Path(__file__).resolve().parents[4] / "CoreCode").exists():
	sys.path.append(
		str(Path(__file__).resolve().parents[4] / "CoreCode"))

tools_path = Path(__file__).resolve().parents[4] / "Tools"

if tools_path.exists():
	if str(tools_path) not in sys.path:
		sys.path.append(str(tools_path))
else:
	raise FileNotFoundError(f"Tools path not found: {tools_path}")

tools_tests_path = tools_path / "tests"

if tools_tests_path.exists():
	if str(tools_tests_path) not in sys.path:
		sys.path.append(str(tools_tests_path))
else:
	raise FileNotFoundError(f"Tools tests path not found: {tools_tests_path}")