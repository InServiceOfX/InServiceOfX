"""Tesseract OCR-based tiled extraction runner.

Drop-in replacement for TiledRunner that uses Tesseract instead of a VLM.
Benefits over VLM: no GPU required, faster (~0.1s/tile vs ~1.3s), much
higher character accuracy at small font sizes.

Each tile is upscaled 3× then passed to Tesseract (PSM 11 = sparse text,
LSTM engine).  Confident words (conf ≥ threshold) are matched against
both dash-separated tag patterns (FCV-001) and no-dash compound patterns
(S2TCV3) via post-processing.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, List

import fitz  # PyMuPDF

from clipdftiledextraction.Core.PDFTiledConfiguration import PDFTiledConfiguration
from clipdftiledextraction.Core.TagMerger import merge_tile_tags
from clipdftiledextraction.Core.TileGenerator import generate_tiles


def _extract_tags_tesseract(
    tile_img: Any,
    min_conf: int = 60,
    upscale: int = 3,
) -> tuple[List[str], str]:
    """Run Tesseract on a PIL tile image and return (tags, raw_text)."""
    import pytesseract
    from PIL import Image, ImageFilter, ImageEnhance

    img = tile_img.convert("L")
    w, h = img.size
    img = img.resize((w * upscale, h * upscale), Image.LANCZOS)
    img = ImageEnhance.Contrast(img).enhance(1.8)
    img = img.filter(ImageFilter.SHARPEN)

    data = pytesseract.image_to_data(
        img,
        config="--psm 11 --oem 1",
        output_type=pytesseract.Output.DICT,
    )
    words = [
        w.strip()
        for w, c in zip(data["text"], data["conf"])
        if w.strip() and int(c) >= min_conf
    ]
    raw = "\n".join(words)

    from clipdftiledextraction.Core.TagMerger import parse_tags_from_response, _parse_compound_tags
    tags = list(dict.fromkeys(
        parse_tags_from_response(raw) + _parse_compound_tags(words)
    ))
    return tags, raw


class TesseractRunner:
    """Extract P&ID tags from PDF tiles using Tesseract OCR (no GPU required)."""

    def __init__(self, configuration: PDFTiledConfiguration):
        self._cfg = configuration

    def run(self) -> None:
        pdf_paths = self._cfg.list_input_pdfs()
        if not pdf_paths:
            print(f"No PDFs found at {self._cfg.input_path}")
            return
        self._cfg.output_path.mkdir(parents=True, exist_ok=True)
        for pdf_path in pdf_paths:
            self._process_pdf(pdf_path)

    def _process_pdf(self, pdf_path: Path) -> None:
        doc_dir = self._cfg.output_path / pdf_path.stem
        doc_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== {pdf_path.name} (Tesseract) ===")

        doc = fitz.open(str(pdf_path))
        manifest_entries = []

        for page_index in range(len(doc)):
            page_num = page_index + 1
            out_path = doc_dir / f"page_{page_num}.json"

            if self._cfg.skip_existing and out_path.exists():
                print(f"  page {page_num}: skipped (exists)")
                manifest_entries.append({"page": page_num, "status": "skipped"})
                continue

            start = time.time()
            result = self._process_page(doc, page_index, page_num, doc_dir)
            elapsed = time.time() - start
            out_path.write_text(json.dumps(result, indent=2))
            n_tags = len(result["merged_tags"])
            print(f"  page {page_num}: {n_tags} tags, {len(result['tiles'])} tiles in {elapsed:.1f}s")
            manifest_entries.append({
                "page": page_num,
                "status": "ok",
                "seconds": elapsed,
                "merged_tag_count": n_tags,
                "output": out_path.name,
            })

        (doc_dir / "manifest.json").write_text(
            json.dumps({
                "pdf_path": str(pdf_path),
                "pdf_stem": pdf_path.stem,
                "num_pages": len(doc),
                "dpi": self._cfg.pdf_dpi,
                "backend": "tesseract",
                "grid": {
                    "cols": self._cfg.grid_cols,
                    "rows": self._cfg.grid_rows,
                    "overlap_fraction": self._cfg.overlap_fraction,
                },
                "pages": manifest_entries,
            }, indent=2)
        )

    def _process_page(self, doc: Any, page_index: int, page_num: int, doc_dir: Path) -> dict:
        from PIL import Image

        page = doc[page_index]
        mat = fitz.Matrix(self._cfg.pdf_dpi / 72, self._cfg.pdf_dpi / 72)
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        if self._cfg.save_intermediate_images:
            img.save(
                str(doc_dir / f"page_{page_num}_tess.{self._cfg.image_format.lower()}"),
                self._cfg.image_format,
            )

        tiles = generate_tiles(
            img,
            grid_cols=self._cfg.grid_cols,
            grid_rows=self._cfg.grid_rows,
            overlap_fraction=self._cfg.overlap_fraction,
        )

        tile_records = []
        tile_responses = []

        for tile in tiles:
            t0 = time.time()
            tags, raw = _extract_tags_tesseract(tile.image)
            tile_seconds = time.time() - t0
            tile_responses.append("\n".join(tags))
            tile_records.append({
                "col": tile.col,
                "row": tile.row,
                "bbox": list(tile.bbox),
                "response": raw,
                "tags": tags,
                "seconds": tile_seconds,
                "backend": "tesseract",
            })

        merged = merge_tile_tags(tile_responses)
        return {
            "page": page_num,
            "dpi": self._cfg.pdf_dpi,
            "backend": "tesseract",
            "grid": {
                "cols": self._cfg.grid_cols,
                "rows": self._cfg.grid_rows,
                "overlap_fraction": self._cfg.overlap_fraction,
            },
            "tiles": tile_records,
            "merged_tags": merged.tags,
            "tile_coverage": merged.tile_coverage,
            "seconds_total": sum(t["seconds"] for t in tile_records),
        }
