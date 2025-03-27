from pathlib import Path
import sys

# To obtain modules from Music
if Path(__file__).resolve().parents[2].exists():
	sys.path.append(str(Path(__file__).resolve().parents[2]))

print(sys.path)

# To obtain modules from CoreCode
if (Path(__file__).resolve().parents[3] / "CoreCode").exists():
	sys.path.append(
		str(Path(__file__).resolve().parents[3] / "CoreCode"))

print(sys.path)
