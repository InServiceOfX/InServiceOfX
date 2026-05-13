import json
import time
from dataclasses import dataclass
from pathlib import Path

from clipdfcolqwenquery.Core.PDFQueryConfiguration import (
    PDFQueryConfiguration,
)


@dataclass(frozen=True)
class IndexedPage:
    pdf: str
    page: int
    embedding_path: Path
    page_image_path: Path | None


class QueryRunner:
    def __init__(
        self,
        embedder,
        query_configuration: PDFQueryConfiguration,
    ):
        self._embedder = embedder
        self._query_configuration = query_configuration

    def run(self) -> None:
        if not self._embedder.is_loaded():
            print("Loading ColQwen2.5 embedder...")
            self._embedder.load()
            print("ColQwen2.5 loaded.")

        pages = self._load_indexed_pages()
        if not pages:
            raise FileNotFoundError(
                f"No indexed page embeddings found under "
                f"{self._query_configuration.index_path}"
            )
        print(f"Loaded {len(pages)} indexed pages.")

        queries = self._query_configuration.queries()
        query_start = time.time()
        query_embeddings = self._embedder.embed_queries(queries)
        print(
            f"Embedded {len(queries)} queries in "
            f"{time.time() - query_start:.1f}s."
        )

        results = {
            "index_path": str(self._query_configuration.index_path),
            "top_k": self._query_configuration.top_k,
            "queries": [],
        }

        for query_index, query_text in enumerate(queries):
            hits = self._score_query(
                query_embedding=query_embeddings[query_index],
                pages=pages,
            )
            results["queries"].append(
                {
                    "query": query_text,
                    "hits": hits[: self._query_configuration.top_k],
                }
            )
            print(f"\nQuery: {query_text}")
            for hit in hits[: self._query_configuration.top_k]:
                print(
                    f"  #{hit['rank']} score={hit['score']:.4f} "
                    f"{hit['pdf']} page {hit['page']}"
                )

        output_file = self._query_configuration.resolved_output_file()
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(json.dumps(results, indent=2))
        print(f"\nWrote {output_file}")

    def _load_indexed_pages(self) -> list[IndexedPage]:
        index_path = self._query_configuration.index_path
        if not index_path.exists():
            raise FileNotFoundError(f"index_path does not exist: {index_path}")

        manifest_paths: list[Path]
        if (index_path / "manifest.json").exists():
            manifest_paths = [index_path / "manifest.json"]
        else:
            manifest_paths = sorted(index_path.glob("*/manifest.json"))

        pages: list[IndexedPage] = []
        for manifest_path in manifest_paths:
            manifest = json.loads(manifest_path.read_text())
            pdf_dir = manifest_path.parent
            pdf_name = manifest.get("pdf_stem", pdf_dir.name)
            for page_record in manifest.get("pages", []):
                embedding_name = page_record.get("embedding")
                if not embedding_name:
                    continue
                embedding_path = pdf_dir / embedding_name
                if not embedding_path.exists():
                    continue
                image_name = page_record.get("image")
                image_path = pdf_dir / image_name if image_name else None
                pages.append(
                    IndexedPage(
                        pdf=pdf_name,
                        page=int(page_record["page"]),
                        embedding_path=embedding_path,
                        page_image_path=image_path
                        if image_path and image_path.exists()
                        else None,
                    )
                )
        return pages

    def _score_query(self, query_embedding, pages: list[IndexedPage]):
        import torch
        from safetensors.torch import load_file

        scored_hits = []
        query_batch = query_embedding.unsqueeze(0)

        for page in pages:
            loaded = load_file(str(page.embedding_path))
            page_embedding = loaded["embedding"].to(query_batch.device)
            page_batch = page_embedding.unsqueeze(0)
            with torch.no_grad():
                score_tensor = self._embedder.score(query_batch, page_batch)
            score = float(score_tensor[0, 0].detach().cpu().item())
            scored_hits.append(
                {
                    "score": score,
                    "pdf": page.pdf,
                    "page": page.page,
                    "embedding": str(page.embedding_path),
                    "page_image": str(page.page_image_path)
                    if page.page_image_path
                    else None,
                }
            )

        scored_hits.sort(key=lambda hit: hit["score"], reverse=True)
        for rank, hit in enumerate(scored_hits, start=1):
            hit["rank"] = rank
        return scored_hits
