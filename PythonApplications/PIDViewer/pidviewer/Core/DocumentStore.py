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
    mineru_element_count: int = 0
    mineru_element_types: List[str] = field(default_factory=list)


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
                    mineru_element_count=element_count,
                    mineru_element_types=element_types,
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

                if snippets:
                    hits.append({
                        "doc_id": doc_id,
                        "page": page_num,
                        "snippets": snippets,
                    })
                    if len(hits) >= max_results:
                        return hits
        return hits
