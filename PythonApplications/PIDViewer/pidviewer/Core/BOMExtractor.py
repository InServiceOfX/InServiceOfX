"""Parse BOM component tables from MinerU HTML output.

Handles P&ID component table schemas including:
- Standard format: Identifier/Subsystem/Assembly/Type/Description/Line Size/Connection Type/MEOP/Manufacturer/Part Number
- Alternate format: Component #/ID/System/Component Type/Description/Fluid Connection/Connection Type/MEOP/Manufacturer/P/N/Component Assembly
  (some tables have a rowspan group cell as first column, so data rows are 1 shorter than header)
"""
from __future__ import annotations

import re
from html.parser import HTMLParser
from typing import Dict, List, Optional

_COLUMN_MAP: dict[str, str] = {
    "identifier": "identifier",
    "id": "identifier",
    "subsystem": "subsystem",
    "system": "subsystem",
    "assembly": "assembly",
    "component assembly": "assembly",
    "type": "component_type",
    "component type": "component_type",
    "description": "description",
    "line size (in)": "line_size",
    "fluid connection [in]": "line_size",
    "fluid connection": "line_size",
    "connection type": "connection_type",
    "system meop (psia)": "meop_psia",
    "meop [psia]": "meop_psia",
    "meop[psia]": "meop_psia",
    "meop (psia)": "meop_psia",
    "manufacturer": "manufacturer",
    "part number": "part_number",
    "p/n": "part_number",
    "manufacturer p/n": "part_number",
    "low [f]": "temp_low_f",
    "high [f]": "temp_high_f",
    "range [psia]": "range_psia",
    "range[psia]": "range_psia",
    "range[unit]": "range_psia",
}

_IDENTIFIER_RE = re.compile(r"^[A-Z][A-Z0-9\-_]{2,24}$")


class _TableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self._row: list[str] = []
        self._cell = ""
        self._in_cell = False

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in ("td", "th"):
            self._in_cell = True
            self._cell = ""
        elif tag == "tr":
            self._row = []

    def handle_endtag(self, tag: str) -> None:
        if tag in ("td", "th"):
            self._row.append(self._cell.strip())
            self._in_cell = False
        elif tag == "tr" and self._row:
            self.rows.append(self._row)
            self._row = []

    def handle_data(self, data: str) -> None:
        if self._in_cell:
            self._cell += data

    def handle_entityref(self, name: str) -> None:
        mapping = {"amp": "&", "lt": "<", "gt": ">", "quot": '"', "apos": "'"}
        if self._in_cell:
            self._cell += mapping.get(name, "")


def _parse_table(html: str) -> list[list[str]]:
    p = _TableParser()
    p.feed(html)
    return p.rows


def extract_bom_entries(elements: list, source_page: int) -> list[dict]:
    """Parse MinerU element list for one page; return BOM entry dicts."""
    entries: list[dict] = []

    for el in elements:
        if el.get("type") != "table":
            continue
        rows = _parse_table(el.get("content", ""))
        if len(rows) < 2:
            continue

        # Find header row: must contain a cell that normalises to "identifier"
        header_idx: Optional[int] = None
        for i, row in enumerate(rows[:4]):
            if any(_COLUMN_MAP.get(c.lower().strip()) == "identifier" for c in row):
                header_idx = i
                break
        if header_idx is None:
            continue

        header = rows[header_idx]
        col_map: dict[int, str] = {
            j: _COLUMN_MAP[c.lower().strip()]
            for j, c in enumerate(header)
            if _COLUMN_MAP.get(c.lower().strip())
        }
        id_cols = [j for j, v in col_map.items() if v == "identifier"]
        if not id_cols:
            continue
        id_col = id_cols[0]

        for row in rows[header_idx + 1:]:
            if not row:
                continue
            offset = max(0, min(len(header) - len(row), 2))
            eff_id = id_col - offset
            if not (0 <= eff_id < len(row)):
                continue
            identifier = row[eff_id].strip().upper()
            if not identifier or not _IDENTIFIER_RE.match(identifier):
                continue

            entry: dict = {"identifier": identifier, "source_page": source_page}
            for j, canonical in col_map.items():
                eff_j = j - offset
                if 0 <= eff_j < len(row):
                    val = row[eff_j].strip()
                    if val and val != identifier:
                        entry.setdefault(canonical, val)
            entries.append(entry)

    return entries
