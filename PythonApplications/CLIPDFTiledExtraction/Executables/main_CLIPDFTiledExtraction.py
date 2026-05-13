"""Tiled P&ID extraction — tiles each page and runs per-tile VLM tag OCR."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(
    0,
    str(
        Path(__file__).resolve().parents[3]
        / "PythonLibraries"
        / "HuggingFace"
        / "MoreMinerU"
    ),
)

from clipdftiledextraction.ApplicationPaths import (
    DEFAULT_PDF_CONFIG_PATH,
    DEFAULT_QWEN_CONFIG_PATH,
)
from clipdftiledextraction.Core.PDFTiledConfiguration import PDFTiledConfiguration
from clipdftiledextraction.Core.TiledRunner import TiledRunner


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tile PDF pages and run per-tile Qwen3-VL tag extraction. "
            "Merges results across tiles and writes per-page JSON."
        )
    )
    parser.add_argument(
        "--pdf-config",
        type=Path,
        default=DEFAULT_PDF_CONFIG_PATH,
        help=f"PDF tiling configuration YAML (default: {DEFAULT_PDF_CONFIG_PATH})",
    )
    parser.add_argument(
        "--qwen-config",
        type=Path,
        default=DEFAULT_QWEN_CONFIG_PATH,
        help=f"Qwen3VL configuration YAML (default: {DEFAULT_QWEN_CONFIG_PATH})",
    )
    parser.add_argument(
        "--currentpath",
        action="store_true",
        help="Resolve config paths relative to the current working directory",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    pdf_config_path = args.pdf_config
    qwen_config_path = args.qwen_config
    if args.currentpath:
        pdf_config_path = Path.cwd() / pdf_config_path.name
        qwen_config_path = Path.cwd() / qwen_config_path.name

    from moremineru.Configurations import Qwen3VLConfiguration
    from moremineru.Applications import Qwen3VLVLLM

    pdf_config = PDFTiledConfiguration.from_yaml(pdf_config_path)
    qwen_config = Qwen3VLConfiguration.from_yaml(qwen_config_path)

    runner = TiledRunner(
        vllm_wrapper=Qwen3VLVLLM(qwen_config),
        configuration=pdf_config,
    )
    runner.run()


if __name__ == "__main__":
    main()
