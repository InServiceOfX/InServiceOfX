"""End-to-end per-page tiled extraction runner.

For each PDF page:
  1. Rasterise at high DPI.
  2. Split into overlapping tiles via TileGenerator.
  3. Run the VLM (Qwen3VLVLLM) on each tile with a strict OCR prompt.
  4. Merge and deduplicate extracted tags via TagMerger.
  5. Write a JSON result file per page.

The JSON output schema per page:
  {
    "page": <int>,
    "dpi": <int>,
    "grid": {"cols": int, "rows": int, "overlap_fraction": float},
    "tiles": [
      {
        "col": int, "row": int,
        "bbox": [left, top, right, bottom],
        "response": "<raw VLM text>",
        "tags": ["<tag>", ...],
        "seconds": float
      },
      ...
    ],
    "merged_tags": ["<tag>", ...],
    "tile_coverage": {"<tag>": [<tile_index>, ...]},
    "seconds_total": float
  }
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF

from clipdftiledextraction.Core.PDFTiledConfiguration import PDFTiledConfiguration
from clipdftiledextraction.Core.TagMerger import merge_tile_tags
from clipdftiledextraction.Core.TileGenerator import generate_tiles


class TiledRunner:
    def __init__(
        self,
        vllm_wrapper: Any,
        configuration: PDFTiledConfiguration,
    ):
        self._vllm = vllm_wrapper
        self._cfg = configuration

    def run(self) -> None:
        if not self._vllm.is_loaded():
            print("Loading VLM…")
            self._vllm.load()
            print("VLM loaded.")

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
        print(f"\n=== {pdf_path.name} ===")

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
            print(
                f"  page {page_num}: {n_tags} tags, "
                f"{len(result['tiles'])} tiles in {elapsed:.1f}s"
            )
            manifest_entries.append({
                "page": page_num,
                "status": "ok",
                "seconds": elapsed,
                "merged_tag_count": n_tags,
                "output": out_path.name,
            })

        (doc_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "pdf_path": str(pdf_path),
                    "pdf_stem": pdf_path.stem,
                    "num_pages": len(doc),
                    "dpi": self._cfg.pdf_dpi,
                    "grid": {
                        "cols": self._cfg.grid_cols,
                        "rows": self._cfg.grid_rows,
                        "overlap_fraction": self._cfg.overlap_fraction,
                    },
                    "pages": manifest_entries,
                },
                indent=2,
            )
        )

    def _process_page(
        self,
        doc: Any,
        page_index: int,
        page_num: int,
        doc_dir: Path,
    ) -> dict:
        from PIL import Image

        page = doc[page_index]
        mat = fitz.Matrix(self._cfg.pdf_dpi / 72, self._cfg.pdf_dpi / 72)
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        if self._cfg.save_intermediate_images:
            img.save(
                str(doc_dir / f"page_{page_num}.{self._cfg.image_format.lower()}"),
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

        for tile_index, tile in enumerate(tiles):
            if self._cfg.save_intermediate_images:
                tile_img_name = f"page_{page_num}_tile_{tile.col}x{tile.row}.{self._cfg.image_format.lower()}"
                tile.image.save(str(doc_dir / tile_img_name), self._cfg.image_format)

            t0 = time.time()
            response = self._vllm.generate(
                tile.image,
                self._cfg.tile_prompt,
                sampling_overrides=self._cfg.sampling_overrides,
            )
            tile_seconds = time.time() - t0

            from clipdftiledextraction.Core.TagMerger import parse_tags_from_response
            tags = parse_tags_from_response(response)
            tile_responses.append(response)

            tile_records.append({
                "col": tile.col,
                "row": tile.row,
                "bbox": list(tile.bbox),
                "response": response,
                "tags": tags,
                "seconds": tile_seconds,
            })

        merged = merge_tile_tags(tile_responses)

        return {
            "page": page_num,
            "dpi": self._cfg.pdf_dpi,
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
