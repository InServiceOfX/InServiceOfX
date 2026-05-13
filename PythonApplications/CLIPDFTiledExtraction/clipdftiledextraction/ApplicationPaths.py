"""Canonical paths within the CLIPDFTiledExtraction application tree."""
from pathlib import Path

_APP_DIR = Path(__file__).resolve().parent.parent

CONFIGURATIONS_DIR = _APP_DIR / "Configurations"
EXECUTABLES_DIR = _APP_DIR / "Executables"

DEFAULT_PDF_CONFIG_PATH = CONFIGURATIONS_DIR / "pdf_tiled_configuration.yml"
DEFAULT_QWEN_CONFIG_PATH = CONFIGURATIONS_DIR / "qwen3vl_configuration.yml"
