# CLIPDFTiledExtraction

Tiled P&ID extraction: rasterizes PDF pages at high DPI, splits each page into
an overlapping grid of tiles, runs Qwen3-VL on each tile with a strict tag-OCR
prompt, and merges deduplicated component/instrument tags across tiles.

## Why tiles?

Full-page VLM inference on a dense P&ID compresses the entire diagram into
~1280 visual patches — too coarse to read small instrument tags. Tiling 3×3
with 15% overlap gives each tile ~400-500 patches at 300 DPI, which is
sufficient for OCR-quality tag reading. Tags near tile edges appear in two or
more tiles and are deduplicated in the merge step.

## Output per page

One JSON file per page:
```json
{
  "page": 3,
  "dpi": 300,
  "grid": {"cols": 3, "rows": 3, "overlap_fraction": 0.15},
  "tiles": [
    {
      "col": 0, "row": 0,
      "bbox": [0, 0, 330, 440],
      "response": "<raw VLM text>",
      "tags": ["PT-001", "FCV-42A"],
      "seconds": 8.3
    },
    ...
  ],
  "merged_tags": ["FCV-42A", "PT-001", "SV-HP-3", ...],
  "tile_coverage": {"PT-001": [0, 3], "FCV-42A": [0], ...},
  "seconds_total": 74.7
}
```

`tile_coverage` maps each tag to the list of tile indices where it was seen.
Tags appearing in only one tile are more likely hallucinations; tags appearing
in two or more tiles that share overlap are more reliable.

## Running inside the container

1. Copy configs:
   ```bash
   cp Configurations/pdf_tiled_configuration.yml.example Configurations/pdf_tiled_configuration.yml
   cp Configurations/qwen3vl_configuration.yml.example Configurations/qwen3vl_configuration.yml
   # Edit paths
   ```

2. Run:
   ```bash
   python Executables/main_CLIPDFTiledExtraction.py --currentpath
   ```

For a 9-page P&ID with a 3×3 grid, expect ~9 tiles × ~9 pages × ~8-10s/tile
= roughly 650-800 seconds total on a 12 GB RTX 3060 with AWQ-8bit Qwen3-VL-4B.

## Tests (no GPU needed)

```bash
pytest tests/ -v
```

Tests cover `TileGenerator` (geometry, overlap, bbox accuracy) and `TagMerger`
(parsing, deduplication, cross-check against reference tags).
