"""Tests for TagMerger — no vLLM/torch needed."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clipdftiledextraction.Core.TagMerger import (
    merge_tile_tags,
    parse_tags_from_response,
)


def test_parse_tags_simple():
    text = "PT-001\nFCV-42A\nSV-HP-3"
    tags = parse_tags_from_response(text)
    assert "PT-001" in tags
    assert "FCV-42A" in tags
    assert "SV-HP-3" in tags


def test_parse_tags_strips_annotation():
    text = "PT-001 - Propellant Tank Pressure\nFCV-42A (fuel control valve)"
    tags = parse_tags_from_response(text)
    assert "PT-001" in tags
    assert "FCV-42A" in tags
    assert all(" " not in t for t in tags)


def test_parse_tags_filters_none_response():
    tags = parse_tags_from_response("NONE")
    assert len(tags) == 0


def test_parse_tags_filters_prose():
    text = (
        "The diagram shows a complex piping layout with multiple valves.\n"
        "PT-042\n"
        "This region contains the pressurization system components."
    )
    tags = parse_tags_from_response(text)
    assert tags == ["PT-042"]


def test_parse_tags_comma_separated():
    text = "PT-001, FCV-42A, SV-HP-3"
    tags = parse_tags_from_response(text)
    assert set(tags) == {"PT-001", "FCV-42A", "SV-HP-3"}


def test_merge_deduplicates():
    responses = [
        "PT-001\nFCV-42A",
        "FCV-42A\nSV-HP-3",
        "PT-001\nSV-HP-3",
    ]
    result = merge_tile_tags(responses)
    assert len(result.tags) == 3
    assert sorted(result.tags) == ["FCV-42A", "PT-001", "SV-HP-3"]


def test_merge_tile_coverage():
    responses = [
        "PT-001\nFCV-42A",
        "FCV-42A\nSV-HP-3",
    ]
    result = merge_tile_tags(responses)
    assert 0 in result.tile_coverage["FCV-42A"]
    assert 1 in result.tile_coverage["FCV-42A"]
    assert result.tile_coverage["PT-001"] == [0]


def test_merge_cross_check():
    # XYZ-999 is a syntactically valid tag but absent from the reference list
    responses = ["PT-001\nFCV-42A\nSV-HP-3\nXYZ-999"]
    reference = ["PT-001", "SV-HP-3"]
    result = merge_tile_tags(responses, reference_tags=reference)
    assert "PT-001" in result.crossed
    assert "SV-HP-3" in result.crossed
    assert "FCV-42A" in result.uncrossed
    assert "XYZ-999" in result.uncrossed


def test_merge_empty_responses():
    result = merge_tile_tags(["NONE", "", "NONE"])
    assert result.tags == []


def test_tag_lowercase_normalised():
    tags = parse_tags_from_response("pt-001\nfcv-42a")
    assert "PT-001" in tags
    assert "FCV-42A" in tags
