"""Reads P&ID topology files produced by extract_pid_topology.py or written manually."""
import json
from pathlib import Path
from typing import Any, Dict, Optional


class TopologyStore:
    def __init__(self, topology_root: Optional[Path]):
        self._root = topology_root

    def _page_dir(self, doc_id: str, page: int) -> Optional[Path]:
        if not self._root:
            return None
        return self._root / doc_id / f"page_{page}"

    def get_topology_json(self, doc_id: str, page: int) -> Optional[Dict[str, Any]]:
        d = self._page_dir(doc_id, page)
        if not d:
            return None
        p = d / "topology.json"
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text())
        except Exception:
            return None

    def get_topology_mermaid(self, doc_id: str, page: int) -> Optional[str]:
        d = self._page_dir(doc_id, page)
        if not d:
            return None
        p = d / "topology.mermaid"
        return p.read_text() if p.exists() else None

    def list_topology_pages(self, doc_id: str) -> list[int]:
        """Return page numbers that have topology data under doc_id."""
        if not self._root:
            return []
        doc_dir = self._root / doc_id
        if not doc_dir.exists():
            return []
        pages = []
        for child in sorted(doc_dir.iterdir()):
            if child.is_dir() and child.name.startswith("page_"):
                try:
                    n = int(child.name.split("_")[1])
                    if (child / "topology.json").exists():
                        pages.append(n)
                except (IndexError, ValueError):
                    pass
        return pages

    def has_topology(self, doc_id: str, page: int) -> bool:
        d = self._page_dir(doc_id, page)
        return bool(d and (d / "topology.json").exists())

    def get_source_image_path(self, doc_id: str, page: int) -> Optional[Path]:
        """Return path to source P&ID image if one was placed alongside topology data."""
        candidates = [
            self._page_dir(doc_id, page),
            self._root / doc_id if self._root else None,
        ]
        for d in candidates:
            if d is None:
                continue
            for name in ("source.png", "source.jpg", "source.jpeg"):
                p = d / name
                if p.exists():
                    return p
        return None
