"""Tests for TileGenerator — no vLLM/torch needed."""
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clipdftiledextraction.Core.TileGenerator import generate_tiles


def make_image(width: int = 900, height: int = 1200) -> Image.Image:
    return Image.new("RGB", (width, height), color=(200, 200, 200))


def test_tile_count_3x3():
    tiles = generate_tiles(make_image(), grid_cols=3, grid_rows=3)
    assert len(tiles) == 9


def test_tile_count_4x4():
    tiles = generate_tiles(make_image(), grid_cols=4, grid_rows=4)
    assert len(tiles) == 16


def test_tile_bbox_stays_within_image():
    img = make_image(900, 1200)
    for tile in generate_tiles(img, grid_cols=3, grid_rows=3, overlap_fraction=0.20):
        left, top, right, bottom = tile.bbox
        assert left >= 0
        assert top >= 0
        assert right <= img.width
        assert bottom <= img.height
        assert right > left
        assert bottom > top


def test_tile_image_matches_bbox():
    img = make_image(900, 1200)
    for tile in generate_tiles(img, grid_cols=3, grid_rows=3):
        left, top, right, bottom = tile.bbox
        expected_w = right - left
        expected_h = bottom - top
        assert tile.image.width == expected_w
        assert tile.image.height == expected_h


def test_adjacent_tiles_overlap():
    """With overlap_fraction > 0, adjacent tiles should share some pixel columns."""
    img = make_image(900, 1200)
    tiles = generate_tiles(img, grid_cols=3, grid_rows=1, overlap_fraction=0.15)
    # tiles are ordered (col=0,row=0), (col=1,row=0), (col=2,row=0)
    right_of_0 = tiles[0].bbox[2]
    left_of_1 = tiles[1].bbox[0]
    assert left_of_1 < right_of_0, "Adjacent tiles should overlap"


def test_grid_index_assignment():
    tiles = generate_tiles(make_image(), grid_cols=3, grid_rows=3)
    col_rows = {(t.col, t.row) for t in tiles}
    expected = {(c, r) for c in range(3) for r in range(3)}
    assert col_rows == expected
