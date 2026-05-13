"""Merge and deduplicate component/instrument tag lists from tiled VLM output.

Each tile produces a block of text. This module:
  1. Splits each response into candidate tag lines.
  2. Normalises (strip whitespace, upper-case).
  3. Deduplicates across tiles.
  4. Optionally cross-checks against a reference list (e.g., from MinerU tables).

Tag heuristics:
  - A "tag" is a short alphanumeric token that looks like an instrument/component
    identifier.  Examples: PT-001, FCV-42A, PG-103, VLOX-1, SV-HP-3.
  - Lines that look like prose (> 60 chars, mostly spaces) are filtered out.
  - Lines that are blank or contain only punctuation are dropped.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Set


_TAG_PATTERN = re.compile(
    # Letter prefix (1-8 chars) followed by one or more dash/underscore segments.
    # Segment length capped at 6 to exclude long English words.
    # Examples: PT-001, FCV-42A, SV-HP-3, VLOX-1, PG-103, S2TYP-01
    r"^[A-Z]{1,8}(?:[-_][A-Z0-9]{1,6})+$"
)

# Matches simple PREFIX-NNN tags (no letter suffixes) used for sequential-run detection.
_SEQUENTIAL_TAG_PATTERN = re.compile(r"^([A-Z]{1,8})-(\d{1,5})$")


def _looks_like_tag(token: str) -> bool:
    """True for all-caps alphanumeric tokens that look like P&ID component tags."""
    token = token.strip().upper()
    if len(token) < 3 or len(token) > 25:
        return False
    if " " in token:
        return False
    # Must contain at least one digit (filters bare letter codes like "NONE")
    if not any(c.isdigit() for c in token):
        return False
    return bool(_TAG_PATTERN.match(token))


def is_sequential_run(tags: List[str]) -> bool:
    """Return True when tags appear to be a hallucinated sequential number series.

    A tile whose entire output is FCV-001, FCV-002, ..., FCV-073 is almost
    certainly fabricated: real P&ID tiles never contain one prefix numbered
    contiguously from 001.  Threshold: ≥5 tags, all matching PREFIX-NNN, with
    at least one prefix whose numbers form a gapless consecutive run.
    """
    if len(tags) < 5:
        return False
    nums_by_prefix: dict[str, list[int]] = {}
    for tag in tags:
        m = _SEQUENTIAL_TAG_PATTERN.match(tag)
        if not m:
            return False  # Mixed-format response — let it pass
        nums_by_prefix.setdefault(m.group(1), []).append(int(m.group(2)))
    for nums in nums_by_prefix.values():
        if len(nums) >= 5:
            s = sorted(nums)
            if s == list(range(s[0], s[0] + len(s))):
                return True
    return False


def parse_tags_from_response(text: str) -> List[str]:
    """Extract candidate tag tokens from a VLM response string.

    The VLM is prompted to return one tag per line.  We extract each non-blank
    line, strip annotation suffixes like "(valve)" or " — description", and
    validate with the tag heuristic.
    """
    tags: List[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        # Strip common annotation suffixes: "(valve)", " - description", " — note"
        line = re.split(r"\s+[-—–:]\s+", line)[0]
        line = re.sub(r"\s*\(.*?\)\s*$", "", line).strip()
        # Allow comma/semicolon-separated lists on a single line
        for candidate in re.split(r"[,;]+", line):
            candidate = candidate.strip()
            if _looks_like_tag(candidate):
                tags.append(candidate.upper())
    return tags


@dataclass
class MergedTagResult:
    tags: List[str]
    tile_coverage: dict  # {tag: [tile_indices_that_saw_it]}
    uncrossed: List[str] = field(default_factory=list)
    crossed: List[str] = field(default_factory=list)


def merge_tile_tags(
    tile_responses: List[str],
    reference_tags: Optional[List[str]] = None,
) -> MergedTagResult:
    """Merge tag lists from multiple tile responses.

    Parameters
    ----------
    tile_responses:
        One string per tile (the raw VLM response for that tile).
    reference_tags:
        Optional list of tags extracted from the legend/table area of the same
        page by MinerU.  When provided, tags in both lists go into `crossed`;
        tags only from tiles go into `uncrossed`.
    """
    coverage: dict[str, List[int]] = {}

    for tile_index, response in enumerate(tile_responses):
        for tag in parse_tags_from_response(response):
            coverage.setdefault(tag, []).append(tile_index)

    # Deduplicated, sorted tag list
    merged_tags = sorted(coverage.keys())

    crossed: List[str] = []
    uncrossed: List[str] = []
    if reference_tags:
        ref_set: Set[str] = {t.upper().strip() for t in reference_tags}
        for tag in merged_tags:
            (crossed if tag in ref_set else uncrossed).append(tag)
    else:
        uncrossed = merged_tags

    return MergedTagResult(
        tags=merged_tags,
        tile_coverage=coverage,
        uncrossed=uncrossed,
        crossed=crossed,
    )
