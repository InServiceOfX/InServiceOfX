from pathlib import Path
import sys

cli_text_to_speech_path = Path(__file__).resolve().parents[1]

# To obtain modules from CoreCode
if cli_text_to_speech_path.exists():
    if str(cli_text_to_speech_path) not in sys.path:
        sys.path.append(str(cli_text_to_speech_path))
else:
    raise FileNotFoundError(
        f"CLITextToSpeech path {cli_text_to_speech_path} not found")