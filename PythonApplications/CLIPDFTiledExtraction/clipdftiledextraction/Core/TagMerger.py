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

# No-dash compound tag pattern: letter + digit + 2-8 letters + optional digits.
# Matches tags like S2SVT, S2TCV3, S2NCV12, S2SVNTP used in some P&ID systems.
_COMPOUND_TAG_PATTERN = re.compile(r"^[A-Z]\d[A-Z]{2,8}\d{0,4}$")

# Matches PREFIX-NNN or PREFIX-NNNA for sequential-run detection.
_SEQUENTIAL_TAG_PATTERN = re.compile(r"^([A-Z]{1,8})-(\d{1,5})$")
# Also matches PREFIX-NNNx where x is an optional constant letter suffix.
_SEQUENTIAL_TAG_PATTERN2 = re.compile(r"^([A-Z]{1,8})-(\d{1,4})([A-Z]*)$")


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

    def _dense_sequential(nums: list[int], min_count: int = 5, density: float = 0.85) -> bool:
        """Return True if nums form a dense near-consecutive run."""
        if len(nums) < min_count:
            return False
        s = sorted(nums)
        span = s[-1] - s[0] + 1
        # Exact consecutive or near-consecutive (≥85% density, no gaps ≥3)
        if s == list(range(s[0], s[0] + len(s))):
            return True
        if span > 0 and len(s) / span >= density:
            # Check that no single gap is absurdly large (< 5× average gap)
            avg_gap = span / len(s)
            max_gap = max(s[i+1] - s[i] for i in range(len(s)-1))
            return max_gap <= max(3, avg_gap * 3)
        return False

    # Attempt to parse all tags via the alphanumeric suffix pattern.
    groups: dict[tuple, list[int]] = {}
    for tag in tags:
        m = _SEQUENTIAL_TAG_PATTERN2.match(tag)
        if not m:
            return False  # Mixed format — let it pass
        key = (m.group(1), m.group(3))  # (prefix, letter_suffix)
        groups.setdefault(key, []).append(int(m.group(2)))

    # Flag if ANY group forms a dense sequential run of ≥5 numbers.
    for nums in groups.values():
        if _dense_sequential(nums):
            return True
    return False


def is_repeat_loop(raw_response: str, tags: List[str]) -> bool:
    """Return True when a tile response looks like a hallucinated repeat loop.

    Two signatures:
    1. The raw response contains any line repeated 3+ times — the model looped.
    2. Every unique tag appears more than once AND there are ≥4 tags total —
       the model cycled through the same short set of tags many times.
    """
    if not raw_response.strip():
        return False
    lines = [l.strip() for l in raw_response.splitlines() if l.strip()]
    if len(lines) >= 6:
        from collections import Counter
        line_counts = Counter(lines)
        if line_counts.most_common(1)[0][1] >= 3:
            return True
    if len(tags) >= 4:
        unique = set(tags)
        if len(unique) < len(tags) / 2:
            return True
    return False


def _parse_compound_tags(words: List[str]) -> List[str]:
    """Extract no-dash compound tags (e.g. S2TCV3, S2SVNTP) from a word list.

    Used by TesseractRunner where individual confident words are already
    separated; we apply the compound pattern directly to each word.
    """
    tags = []
    for w in words:
        token = w.strip().upper()
        if len(token) >= 4 and _COMPOUND_TAG_PATTERN.match(token):
            tags.append(token)
    return tags


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
