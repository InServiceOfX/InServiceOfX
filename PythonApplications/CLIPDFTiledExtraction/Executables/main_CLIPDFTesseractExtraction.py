"""Tesseract OCR tiled P&ID tag extraction.

Identical pipeline to main_CLIPDFTiledExtraction.py but uses Tesseract
instead of a VLM — no GPU required, faster and more accurate at small fonts.

Usage:
  python3 main_CLIPDFTesseractExtraction.py \
      --pdf-config Configurations/pdf_tiled_configuration.yml \
      [--output-path /path/to/output]
"""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "PythonApplications" / "CLIPDFTiledExtraction"))
sys.path.insert(0, str(REPO_ROOT / "PythonLibraries" / "HuggingFace" / "MoreMinerU"))

from clipdftiledextraction.Core.PDFTiledConfiguration import PDFTiledConfiguration
from clipdftiledextraction.Core.TesseractRunner import TesseractRunner


def main() -> None:
    parser = argparse.ArgumentParser(description="Tesseract tiled P&ID tag extraction")
    parser.add_argument("--pdf-config", required=True, help="Path to pdf_tiled_configuration.yml")
    parser.add_argument(
        "--output-path",
        default=None,
        help="Override output_path from config (for separate Tesseract output dir)",
    )
    args = parser.parse_args()

    cfg = PDFTiledConfiguration.from_yaml(Path(args.pdf_config))
    if args.output_path:
        cfg = cfg.model_copy(update={"output_path": Path(args.output_path)})

    runner = TesseractRunner(configuration=cfg)
    runner.run()


if __name__ == "__main__":
    main()
