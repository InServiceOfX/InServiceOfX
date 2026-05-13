"""Tile a PIL Image into overlapping rectangular regions.

Strategy: divide the image into an (rows × cols) grid; each tile is expanded
by `overlap_fraction` on each side so that symbols near tile edges appear in at
least two tiles and are not clipped.

Returned tiles include their (col, row) grid index and the pixel bounding box
(left, top, right, bottom) in the original image coordinate system, so results
can be spatially placed back on the page.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Tuple

from PIL import Image


@dataclass(frozen=True)
class Tile:
    col: int
    row: int
    bbox: Tuple[int, int, int, int]  # (left, top, right, bottom) in original px
    image: Image.Image


def generate_tiles(
    image: Image.Image,
    grid_cols: int = 3,
    grid_rows: int = 3,
    overlap_fraction: float = 0.15,
) -> list[Tile]:
    """Return a list of overlapping Tile objects covering *image*.

    Parameters
    ----------
    image:
        Source image (any mode, any size).
    grid_cols / grid_rows:
        Number of tile columns / rows.  3×3 = 9 base tiles works well for a
        full-page P&ID with dense instrument lines.
    overlap_fraction:
        Fraction of each base cell width / height to expand per side.
        0.15 means each tile is 30% wider and 30% taller than the base cell,
        so adjacent tiles share ~30% of their area.
    """
    w, h = image.size
    cell_w = w / grid_cols
    cell_h = h / grid_rows
    pad_x = cell_w * overlap_fraction
    pad_y = cell_h * overlap_fraction

    tiles: list[Tile] = []
    for row in range(grid_rows):
        for col in range(grid_cols):
            left = max(0, int(col * cell_w - pad_x))
            top = max(0, int(row * cell_h - pad_y))
            right = min(w, int((col + 1) * cell_w + pad_x))
            bottom = min(h, int((row + 1) * cell_h + pad_y))
            crop = image.crop((left, top, right, bottom))
            tiles.append(
                Tile(
                    col=col,
                    row=row,
                    bbox=(left, top, right, bottom),
                    image=crop,
                )
            )
    return tiles
