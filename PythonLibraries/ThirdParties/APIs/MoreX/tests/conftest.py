from pathlib import Path
import sys

# To obtain modules from MoreX
morex_path = Path(__file__).resolve().parents[1]

if morex_path.exists() and str(morex_path) not in sys.path:
	sys.path.append(str(morex_path))

corecode_path = Path(__file__).resolve().parents[4] / "CoreCode"
if corecode_path.exists() and str(corecode_path) not in sys.path:
	sys.path.append(str(corecode_path))