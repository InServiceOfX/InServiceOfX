"""ColQwen2.5 query REST server.

Loads the ColQwen2.5 model and a pre-built index (produced by
CLIPDFColQwenIndexer) once at startup, then serves ranked-page queries.

Usage (inside Docker):
    cd /InServiceOfX/PythonApplications/CLIPDFColQwenQuery
    PYTHONPATH=/InServiceOfX/PythonLibraries/HuggingFace/MoreMinerU \
    python3 Executables/main_ColQwenQueryServer.py --currentpath

Endpoints:
    GET  /health          — server status + index summary
    GET  /index/list      — list indexed PDF stems and page counts
    POST /query           — rank pages for a text query
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Put the app package on sys.path
_APP_ROOT = Path(__file__).resolve().parents[1]
if str(_APP_ROOT) not in sys.path:
    sys.path.insert(0, str(_APP_ROOT))

# Add MoreMinerU library to sys.path so moremineru is importable
_MOREMINERU = _APP_ROOT.parents[1] / "PythonLibraries" / "HuggingFace" / "MoreMinerU"
if str(_MOREMINERU) not in sys.path:
    sys.path.insert(0, str(_MOREMINERU))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ColQwen2.5 query REST server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--colqwen-config", type=Path, default=None)
    parser.add_argument("--index-path", type=Path, default=None)
    parser.add_argument("--currentpath", action="store_true",
                        help="Resolve config paths relative to cwd")
    return parser.parse_args()


def _default_config_path(name: str, currentpath: bool) -> Path:
    base = Path.cwd() if currentpath else _APP_ROOT / "Configurations"
    return base / name


def main() -> None:
    args = _parse_args()

    colqwen_config_path = args.colqwen_config or _default_config_path(
        "colqwen2_5_configuration.yml", args.currentpath
    )
    if not colqwen_config_path.exists():
        print(f"ERROR: colqwen config not found: {colqwen_config_path}", file=sys.stderr)
        sys.exit(1)

    # Load ColQwen config
    from moremineru.Configurations import ColQwen2_5Configuration
    colqwen_cfg = ColQwen2_5Configuration.from_yaml(colqwen_config_path, validate_paths=True)

    # Resolve index path (default: container-mounted CLIPDFColQwenIndexer output)
    index_path = args.index_path or Path("/Workspace/Generated/CLIPDFColQwenIndexer")

    import uvicorn
    from contextlib import asynccontextmanager
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel as PydanticModel

    # --- State shared across requests ---
    state: Dict[str, Any] = {
        "embedder": None,
        "pages": [],          # list of IndexedPage-like dicts
        "model_name": str(colqwen_cfg.model_path),
        "index_path": str(index_path),
        "status": "loading",
    }

    def _load_index(idx_path: Path) -> list:
        if not idx_path.exists():
            return []
        if (idx_path / "manifest.json").exists():
            manifests = [idx_path / "manifest.json"]
        else:
            manifests = sorted(idx_path.glob("*/manifest.json"))
        pages = []
        for mp in manifests:
            manifest = json.loads(mp.read_text())
            pdf_dir = mp.parent
            pdf_name = manifest.get("pdf_stem", pdf_dir.name)
            for rec in manifest.get("pages", []):
                emb_name = rec.get("embedding")
                if not emb_name:
                    continue
                emb_path = pdf_dir / emb_name
                if not emb_path.exists():
                    continue
                img_name = rec.get("image")
                pages.append({
                    "pdf": pdf_name,
                    "page": int(rec["page"]),
                    "embedding_path": str(emb_path),
                    "image_path": str(pdf_dir / img_name) if img_name else None,
                })
        return pages

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        print("Loading ColQwen2.5 model…")
        from moremineru.Applications import ColQwen2_5Embedder
        embedder = ColQwen2_5Embedder(colqwen_cfg)
        embedder.load()
        print("Model loaded.")

        pages = _load_index(index_path)
        print(f"Loaded {len(pages)} indexed pages from {index_path}")

        state["embedder"] = embedder
        state["pages"] = pages
        state["status"] = "ready"
        yield
        state["status"] = "shutdown"

    app = FastAPI(title="ColQwen2.5 Query Server", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health():
        return {
            "status": state["status"],
            "model": state["model_name"],
            "index_path": state["index_path"],
            "indexed_pages": len(state["pages"]),
            "pdfs": sorted({p["pdf"] for p in state["pages"]}),
        }

    @app.get("/index/list")
    def index_list():
        summary: Dict[str, int] = {}
        for p in state["pages"]:
            summary[p["pdf"]] = summary.get(p["pdf"], 0) + 1
        return {"pdfs": [{"name": k, "pages": v} for k, v in sorted(summary.items())]}

    class QueryRequest(PydanticModel):
        query: str
        top_k: int = 5

    @app.post("/query")
    def query_endpoint(req: QueryRequest):
        import torch
        from safetensors.torch import load_file

        if state["status"] != "ready":
            from fastapi import HTTPException
            raise HTTPException(503, detail=f"Server not ready: {state['status']}")
        if not req.query.strip():
            from fastapi import HTTPException
            raise HTTPException(400, detail="query must not be empty")

        embedder = state["embedder"]
        query_embeddings = embedder.embed_queries([req.query])
        query_batch = query_embeddings[0].unsqueeze(0)

        scored: List[Dict[str, Any]] = []
        for page in state["pages"]:
            loaded = load_file(page["embedding_path"])
            page_emb = loaded["embedding"].to(query_batch.device)
            page_batch = page_emb.unsqueeze(0)
            with torch.no_grad():
                score_val = float(embedder.score(query_batch, page_batch)[0, 0].cpu().item())
            scored.append({
                "pdf": page["pdf"],
                "page": page["page"],
                "score": score_val,
                "image_path": page["image_path"],
            })

        scored.sort(key=lambda h: h["score"], reverse=True)
        hits = []
        for rank, h in enumerate(scored[: req.top_k], start=1):
            h["rank"] = rank
            hits.append(h)

        return {"query": req.query, "top_k": req.top_k, "hits": hits}

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
