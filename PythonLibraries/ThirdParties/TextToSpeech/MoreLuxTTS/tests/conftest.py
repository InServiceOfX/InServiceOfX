from pathlib import Path
import sys

corecode_path = Path(__file__).resolve().parents[4] / "CoreCode"

if corecode_path.exists() and corecode_path not in sys.path:
	sys.path.append(str(corecode_path))
elif not corecode_path.exists():
	raise FileNotFoundError(f"CoreCode path {corecode_path} not found")

from corecode.FileIO.get_project_directory_path import get_project_directory_path

project_directory_path = get_project_directory_path()
lux_tts_path = project_directory_path.parent / "ThirdParty" / "LuxTTS"

if lux_tts_path.exists() and lux_tts_path not in sys.path:
	sys.path.append(str(lux_tts_path))
elif not lux_tts_path.exists():
	raise FileNotFoundError(f"LuxTTS path {lux_tts_path} not found")