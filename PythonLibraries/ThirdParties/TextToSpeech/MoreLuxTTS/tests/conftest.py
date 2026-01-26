from pathlib import Path
import sys

corecode_path = Path(__file__).resolve().parents[4] / "CoreCode"

if corecode_path.exists() and corecode_path not in sys.path:
	sys.path.append(str(corecode_path))
elif not corecode_path.exists():
	raise FileNotFoundError(f"CoreCode path {corecode_path} not found")