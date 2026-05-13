"""Tesseract OCR tag extraction from pre-rasterised MinerU images.

Runs Tesseract on the page_N.png files produced by CLIPDFExtraction,
so no source PDFs are needed.  Useful for processing documents whose
PDFs are only accessible inside Docker.

Usage:
  python3 main_CLIPDFTesseractFromImages.py \
      --pdf-config Configurations/pdf_tiled_configuration.yml \
      --mineru-output /path/to/CLIPDFExtraction \
      --output-path /path/to/CLIPDFTesseractExtraction \
      [--doc DOCUMENT_ID]
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
    parser = argparse.ArgumentParser(
        description="Tesseract tiled extraction from pre-rendered MinerU PNG images"
    )
    parser.add_argument("--pdf-config", required=True, help="Path to pdf_tiled_configuration.yml")
    parser.add_argument("--mineru-output", required=True, help="CLIPDFExtraction output directory")
    parser.add_argument("--output-path", required=True, help="Tesseract output directory")
    parser.add_argument("--doc", default=None, help="Process only this document ID (subdir name)")
    parser.add_argument("--no-skip", action="store_true", help="Re-process pages that already have output (forces bbox data refresh)")
    args = parser.parse_args()

    cfg = PDFTiledConfiguration.from_yaml(Path(args.pdf_config))
    runner = TesseractRunner(configuration=cfg)
    runner.run_from_mineru_images(
        mineru_output_path=Path(args.mineru_output),
        output_path=Path(args.output_path),
        doc_filter=args.doc,
        skip_existing=not args.no_skip,
    )


if __name__ == "__main__":
    main()
