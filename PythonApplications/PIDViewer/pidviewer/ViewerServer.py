"""FastAPI application for the P&ID extraction viewer."""
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from pidviewer.Core.DocumentStore import DocumentStore
from pidviewer.ViewerConfiguration import ViewerConfiguration

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


def create_app(configuration: ViewerConfiguration) -> FastAPI:
    store = DocumentStore(configuration)
    app = FastAPI(title="P&ID Viewer", version="0.1.0")

    # --- Response models ---

    class PageSummary(BaseModel):
        page: int
        status: str
        seconds: float
        has_image: bool
        has_mineru: bool
        has_qwen3vl: bool
        has_tiled: bool
        has_colqwen: bool
        mineru_element_count: int
        mineru_element_types: List[str]
        tiled_tag_count: int = 0
        tiled_hallucinated_tiles: int = 0

    class DocumentSummary(BaseModel):
        doc_id: str
        num_pages: int
        dpi: int
        pdf_path: str
        pages: List[PageSummary]

    # --- Routes ---

    @app.get("/", response_class=HTMLResponse)
    async def index():
        html_path = STATIC_DIR / "index.html"
        if not html_path.exists():
            raise HTTPException(500, "index.html not found")
        return HTMLResponse(content=html_path.read_text())

    @app.get("/api/documents", response_model=List[str])
    async def list_documents():
        return store.list_documents()

    @app.get("/api/documents/{doc_id}", response_model=DocumentSummary)
    async def get_document(doc_id: str):
        doc = store.get_document(doc_id)
        if doc is None:
            raise HTTPException(404, f"Document not found: {doc_id}")
        return DocumentSummary(
            doc_id=doc.doc_id,
            num_pages=doc.num_pages,
            dpi=doc.dpi,
            pdf_path=doc.pdf_path,
            pages=[
                PageSummary(
                    page=p.page,
                    status=p.status,
                    seconds=p.seconds,
                    has_image=p.has_image,
                    has_mineru=p.has_mineru,
                    has_qwen3vl=p.has_qwen3vl,
                    has_tiled=p.has_tiled,
                    has_colqwen=p.has_colqwen,
                    mineru_element_count=p.mineru_element_count,
                    mineru_element_types=p.mineru_element_types,
                    tiled_tag_count=p.tiled_tag_count,
                    tiled_hallucinated_tiles=p.tiled_hallucinated_tiles,
                )
                for p in doc.pages
            ],
        )

    @app.get("/api/documents/{doc_id}/pages/{page}/image")
    async def get_page_image(doc_id: str, page: int):
        image_path = store.get_page_image_path(doc_id, page)
        if image_path is None:
            raise HTTPException(404, f"Image not found for {doc_id} page {page}")
        return FileResponse(str(image_path), media_type="image/png")

    @app.get("/api/documents/{doc_id}/pages/{page}/mineru")
    async def get_page_mineru(doc_id: str, page: int):
        data = store.get_page_mineru(doc_id, page)
        if data is None:
            raise HTTPException(404, f"MinerU output not found for {doc_id} page {page}")
        return data

    @app.get("/api/documents/{doc_id}/pages/{page}/qwen3vl")
    async def get_page_qwen3vl(doc_id: str, page: int):
        text = store.get_page_qwen3vl(doc_id, page)
        if text is None:
            raise HTTPException(404, f"Qwen3VL output not found for {doc_id} page {page}")
        return {"text": text}

    @app.get("/api/documents/{doc_id}/pages/{page}/tiled")
    async def get_page_tiled(doc_id: str, page: int):
        data = store.get_page_tiled(doc_id, page)
        if data is None:
            raise HTTPException(404, f"Tiled output not found for {doc_id} page {page}")
        return data

    @app.get("/api/search")
    async def search(
        q: str = Query(..., description="Search query (case-insensitive substring)"),
        doc: Optional[str] = Query(None, description="Limit to one document"),
        max_results: int = Query(50, ge=1, le=200),
    ):
        if not q.strip():
            raise HTTPException(400, "Query must not be empty")
        hits = store.search(q.strip(), doc_filter=doc, max_results=max_results)
        return {"query": q, "hit_count": len(hits), "hits": hits}

    class ColQwenQueryRequest(BaseModel):
        query: str
        top_k: int = 5

    @app.post("/api/colqwen/query")
    async def colqwen_query(req: ColQwenQueryRequest):
        if not configuration.colqwen_server_url:
            raise HTTPException(503, "colqwen_server_url not configured")
        url = configuration.colqwen_server_url.rstrip("/") + "/query"
        payload = json.dumps({"query": req.query, "top_k": req.top_k}).encode()
        http_req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(http_req, timeout=30) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")
            raise HTTPException(exc.code, detail=detail) from exc
        except Exception as exc:
            raise HTTPException(502, f"ColQwen server error: {exc}") from exc

    @app.get("/api/colqwen/health")
    async def colqwen_health():
        if not configuration.colqwen_server_url:
            return {"available": False, "reason": "colqwen_server_url not configured"}
        url = configuration.colqwen_server_url.rstrip("/") + "/health"
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                data = json.loads(resp.read())
                data["available"] = True
                return data
        except Exception as exc:
            return {"available": False, "reason": str(exc)}

    @app.get("/api/status")
    async def status():
        docs = store.list_documents()
        return {
            "mineru_output_path": str(configuration.mineru_output_path),
            "qwen3vl_output_path": str(configuration.qwen3vl_output_path)
            if configuration.qwen3vl_output_path
            else None,
            "colqwen_index_path": str(configuration.colqwen_index_path)
            if configuration.colqwen_index_path
            else None,
            "colqwen_server_url": configuration.colqwen_server_url,
            "document_count": len(docs),
            "documents": docs,
        }

    return app
