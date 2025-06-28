from pathlib import Path
import sys

tests_path = Path(__file__).resolve().parent

if str(tests_path) not in sys.path:
    sys.path.append(str(tests_path))

project_path = Path(__file__).resolve().parents[1]

if str(project_path) not in sys.path:
    sys.path.append(str(project_path))
