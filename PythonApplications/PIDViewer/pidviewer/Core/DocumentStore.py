"""Reads extraction outputs produced by CLIPDFExtraction and CLIPDFQwen3VLChat."""
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from pidviewer.ViewerConfiguration import ViewerConfiguration


@dataclass
class PageRecord:
    page: int
    status: str
    seconds: float
    has_image: bool
    has_mineru: bool
    has_qwen3vl: bool
    has_colqwen: bool
    has_tiled: bool = False
    has_tesseract: bool = False
    mineru_element_count: int = 0
    mineru_element_types: List[str] = field(default_factory=list)
    tiled_tag_count: int = 0
    tiled_hallucinated_tiles: int = 0
    tesseract_tag_count: int = 0


@dataclass
class DocumentRecord:
    doc_id: str
    num_pages: int
    dpi: int
    pdf_path: str
    pages: List[PageRecord]


class DocumentStore:
    def __init__(self, configuration: ViewerConfiguration):
        self._cfg = configuration

    def list_documents(self) -> List[str]:
        base = self._cfg.mineru_output_path
        if not base.exists():
            return []
        docs = []
        for subdir in sorted(base.iterdir()):
            if subdir.is_dir() and (subdir / "manifest.json").exists():
                docs.append(subdir.name)
        return docs

    def get_document(self, doc_id: str) -> Optional[DocumentRecord]:
        manifest_path = self._cfg.mineru_output_path / doc_id / "manifest.json"
        if not manifest_path.exists():
            return None
        manifest = json.loads(manifest_path.read_text())

        pages = []
        for page_entry in manifest.get("pages", []):
            page_num = page_entry["page"]
            doc_dir = self._cfg.mineru_output_path / doc_id
            image_path = doc_dir / f"page_{page_num}.png"
            mineru_path = doc_dir / f"page_{page_num}.md"

            qwen_path = None
            if self._cfg.qwen3vl_output_path:
                qwen_dir = self._cfg.qwen3vl_output_path / doc_id
                qwen_path = qwen_dir / f"page_{page_num}.txt"

            tiled_path = None
            if self._cfg.tiled_output_path:
                tiled_dir = self._cfg.tiled_output_path / doc_id
                tiled_path = tiled_dir / f"page_{page_num}.json"

            tesseract_path = None
            if self._cfg.tesseract_output_path:
                tess_dir = self._cfg.tesseract_output_path / doc_id
                tesseract_path = tess_dir / f"page_{page_num}.json"

            colqwen_path = None
            if self._cfg.colqwen_index_path:
                cq_dir = self._cfg.colqwen_index_path / doc_id
                colqwen_path = cq_dir / f"page_{page_num}.safetensors"

            element_count = 0
            element_types: List[str] = []
            if mineru_path.exists():
                try:
                    elements = json.loads(mineru_path.read_text())
                    if isinstance(elements, list):
                        element_count = len(elements)
                        types_seen = set()
                        for el in elements:
                            t = el.get("type", "unknown")
                            types_seen.add(t)
                        element_types = sorted(types_seen)
                except Exception:
                    pass

            tiled_tag_count = 0
            tiled_hallucinated_tiles = 0
            if tiled_path and tiled_path.exists():
                try:
                    tiled_data = json.loads(tiled_path.read_text())
                    tiled_tag_count = len(tiled_data.get("merged_tags", []))
                    tiled_hallucinated_tiles = sum(
                        1 for t in tiled_data.get("tiles", [])
                        if t.get("hallucination_suspected")
                    )
                except Exception:
                    pass

            tesseract_tag_count = 0
            if tesseract_path and tesseract_path.exists():
                try:
                    tess_data = json.loads(tesseract_path.read_text())
                    tesseract_tag_count = len(tess_data.get("merged_tags", []))
                except Exception:
                    pass

            pages.append(
                PageRecord(
                    page=page_num,
                    status=page_entry.get("status", "unknown"),
                    seconds=page_entry.get("seconds", 0.0),
                    has_image=image_path.exists(),
                    has_mineru=mineru_path.exists(),
                    has_qwen3vl=bool(qwen_path and qwen_path.exists()),
                    has_colqwen=bool(colqwen_path and colqwen_path.exists()),
                    has_tiled=bool(tiled_path and tiled_path.exists()),
                    has_tesseract=bool(tesseract_path and tesseract_path.exists()),
                    mineru_element_count=element_count,
                    mineru_element_types=element_types,
                    tiled_tag_count=tiled_tag_count,
                    tiled_hallucinated_tiles=tiled_hallucinated_tiles,
                    tesseract_tag_count=tesseract_tag_count,
                )
            )

        return DocumentRecord(
            doc_id=doc_id,
            num_pages=manifest.get("num_pages", len(pages)),
            dpi=manifest.get("dpi", 0),
            pdf_path=manifest.get("pdf_path", ""),
            pages=pages,
        )

    def get_page_image_path(self, doc_id: str, page: int) -> Optional[Path]:
        p = self._cfg.mineru_output_path / doc_id / f"page_{page}.png"
        return p if p.exists() else None

    def get_tile_image_path(self, doc_id: str, page: int, col: int, row: int) -> Optional[Path]:
        if not self._cfg.tiled_output_path:
            return None
        p = self._cfg.tiled_output_path / doc_id / f"page_{page}_tile_{col}x{row}.png"
        return p if p.exists() else None

    def get_page_mineru(self, doc_id: str, page: int) -> Optional[Any]:
        p = self._cfg.mineru_output_path / doc_id / f"page_{page}.md"
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text())
        except Exception:
            return p.read_text()

    def get_page_qwen3vl(self, doc_id: str, page: int) -> Optional[str]:
        if not self._cfg.qwen3vl_output_path:
            return None
        p = self._cfg.qwen3vl_output_path / doc_id / f"page_{page}.txt"
        return p.read_text() if p.exists() else None

    def get_page_tiled(self, doc_id: str, page: int) -> Optional[Any]:
        if not self._cfg.tiled_output_path:
            return None
        p = self._cfg.tiled_output_path / doc_id / f"page_{page}.json"
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text())
        except Exception:
            return None

    def get_page_tesseract(self, doc_id: str, page: int) -> Optional[Any]:
        if not self._cfg.tesseract_output_path:
            return None
        p = self._cfg.tesseract_output_path / doc_id / f"page_{page}.json"
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text())
        except Exception:
            return None

    def get_page_tesseract_tag_bboxes(self, doc_id: str, page: int) -> Optional[Dict]:
        """Aggregate per-tile Tesseract bboxes into page-normalised (0–1) coordinates.

        Returns None if no Tesseract data exists, or a dict with keys:
          has_bboxes, page_w, page_h, tags: {tag: [{x,y,w,h}]}
        """
        data = self.get_page_tesseract(doc_id, page)
        if not data:
            return None

        page_w = data.get("page_w")
        page_h = data.get("page_h")
        if not page_w or not page_h:
            # Fallback: infer from last tile bbox (right, bottom of last tile ≈ page edge)
            tiles = data.get("tiles", [])
            if tiles:
                last_bbox = tiles[-1].get("bbox", [0, 0, 0, 0])
                page_w, page_h = last_bbox[2], last_bbox[3]
        if not page_w or not page_h:
            return None

        result: Dict[str, list] = {}
        seen: set = set()

        for tile in data.get("tiles", []):
            tile_bbox = tile.get("bbox", [0, 0, 0, 0])
            tile_left, tile_top = tile_bbox[0], tile_bbox[1]
            for tag, bboxes in tile.get("tag_bboxes", {}).items():
                for bb in bboxes:
                    px = tile_left + bb["x"]
                    py = tile_top + bb["y"]
                    # Deduplicate detections from overlapping tiles (20px tolerance)
                    key = (tag, round(px / 20), round(py / 20))
                    if key in seen:
                        continue
                    seen.add(key)
                    result.setdefault(tag, []).append({
                        "x": round(px / page_w, 5),
                        "y": round(py / page_h, 5),
                        "w": round(bb["w"] / page_w, 5),
                        "h": round(bb["h"] / page_h, 5),
                    })

        return {
            "has_bboxes": bool(result),
            "page_w": page_w,
            "page_h": page_h,
            "tags": result,
        }

    def get_bom(self, doc_id: str) -> Dict[str, Dict]:
        """Return {identifier: entry_dict} by scanning all MinerU page tables.

        If the same identifier appears on multiple pages, the later page wins
        (later BOM pages tend to be more complete for multi-page documents).
        """
        from pidviewer.Core.BOMExtractor import extract_bom_entries

        bom: Dict[str, Dict] = {}
        doc_dir = self._cfg.mineru_output_path / doc_id
        manifest_path = doc_dir / "manifest.json"
        if not manifest_path.exists():
            return {}
        try:
            manifest = json.loads(manifest_path.read_text())
        except Exception:
            return {}

        for page_entry in manifest.get("pages", []):
            page_num = page_entry["page"]
            md_path = doc_dir / f"page_{page_num}.md"
            if not md_path.exists():
                continue
            try:
                elements = json.loads(md_path.read_text())
            except Exception:
                continue
            for entry in extract_bom_entries(elements, page_num):
                ident = entry["identifier"]
                existing = bom.get(ident)
                if existing is None or existing.get("source_page", 0) < page_num:
                    bom[ident] = entry
        return bom

    def get_colqwen_index_for_document(self, doc_id: str) -> Optional[Path]:
        """Returns manifest.json path for a ColQwen-indexed document, if present."""
        if not self._cfg.colqwen_index_path:
            return None
        p = self._cfg.colqwen_index_path / doc_id / "manifest.json"
        return p if p.exists() else None

    def search(
        self,
        query: str,
        doc_filter: Optional[str] = None,
        max_results: int = 50,
    ) -> List[Dict]:
        """Case-insensitive substring search across MinerU and Qwen3VL text."""
        q = query.lower()
        hits: List[Dict] = []

        docs = [doc_filter] if doc_filter else self.list_documents()
        for doc_id in docs:
            doc_dir = self._cfg.mineru_output_path / doc_id
            manifest_path = doc_dir / "manifest.json"
            if not manifest_path.exists():
                continue
            manifest = json.loads(manifest_path.read_text())
            for page_entry in manifest.get("pages", []):
                page_num = page_entry["page"]
                snippets: List[Dict] = []

                mineru_path = doc_dir / f"page_{page_num}.md"
                if mineru_path.exists():
                    try:
                        elements = json.loads(mineru_path.read_text())
                        for el in (elements if isinstance(elements, list) else []):
                            content = el.get("content", "")
                            if isinstance(content, str) and q in content.lower():
                                idx = content.lower().find(q)
                                start = max(0, idx - 60)
                                end = min(len(content), idx + len(query) + 60)
                                snippets.append({
                                    "source": "mineru",
                                    "type": el.get("type", "unknown"),
                                    "snippet": content[start:end],
                                })
                    except Exception:
                        pass

                if self._cfg.qwen3vl_output_path:
                    qp = self._cfg.qwen3vl_output_path / doc_id / f"page_{page_num}.txt"
                    if qp.exists():
                        text = qp.read_text()
                        if q in text.lower():
                            idx = text.lower().find(q)
                            start = max(0, idx - 80)
                            end = min(len(text), idx + len(query) + 80)
                            snippets.append({
                                "source": "qwen3vl",
                                "type": "text",
                                "snippet": text[start:end],
                            })

                if self._cfg.tiled_output_path:
                    tp = self._cfg.tiled_output_path / doc_id / f"page_{page_num}.json"
                    if tp.exists():
                        try:
                            tiled = json.loads(tp.read_text())
                            matched = [
                                t for t in tiled.get("merged_tags", [])
                                if q in t.lower()
                            ]
                            if matched:
                                snippets.append({
                                    "source": "tiled",
                                    "type": "tags",
                                    "snippet": " ".join(matched[:20]),
                                })
                        except Exception:
                            pass

                if snippets:
                    hits.append({
                        "doc_id": doc_id,
                        "page": page_num,
                        "snippets": snippets,
                    })
                    if len(hits) >= max_results:
                        return hits
        return hits
