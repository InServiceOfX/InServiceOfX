import json
import time
from pathlib import Path

from clipdfcolqwenindexer.Core.PDFIndexConfiguration import (
    PDFIndexConfiguration,
)
from clipdfcolqwenindexer.Core.PDFRasterizer import PDFRasterizer


class IndexRunner:
    def __init__(
        self,
        embedder,
        pdf_configuration: PDFIndexConfiguration,
    ):
        self._embedder = embedder
        self._pdf_configuration = pdf_configuration
        self._rasterizer = PDFRasterizer(dpi=pdf_configuration.pdf_dpi)

    def run(self) -> None:
        if not self._embedder.is_loaded():
            print("Loading ColQwen2.5 embedder...")
            self._embedder.load()
            print("ColQwen2.5 loaded.")

        pdf_paths = self._pdf_configuration.list_input_pdfs()
        if not pdf_paths:
            print(
                f"No PDFs found at {self._pdf_configuration.input_path}; "
                "nothing to do."
            )
            return

        self._pdf_configuration.output_path.mkdir(parents=True, exist_ok=True)

        for pdf_path in pdf_paths:
            self._process_pdf(pdf_path)

    def _process_pdf(self, pdf_path: Path) -> None:
        pdf_output_dir = self._pdf_configuration.output_path / pdf_path.stem
        pdf_output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== {pdf_path.name} ===")
        rasterize_start = time.time()
        pages = list(self._rasterizer.iter_pages(pdf_path))
        rasterize_seconds = time.time() - rasterize_start
        print(
            f"  rasterized {len(pages)} pages in {rasterize_seconds:.1f}s "
            f"@ dpi={self._pdf_configuration.pdf_dpi}"
        )

        manifest = {
            "pdf_path": str(pdf_path),
            "pdf_stem": pdf_path.stem,
            "num_pages": len(pages),
            "dpi": self._pdf_configuration.pdf_dpi,
            "model_path": str(self._embedder._configuration.model_path),
            "embeddings_format": self._pdf_configuration.embeddings_format,
            "pages": [],
        }

        for page_index, image in pages:
            embedding_path = pdf_output_dir / f"page_{page_index}.safetensors"
            image_path = (
                pdf_output_dir
                / f"page_{page_index}."
                f"{self._pdf_configuration.image_format.lower()}"
            )

            if (
                self._pdf_configuration.skip_existing
                and embedding_path.exists()
            ):
                print(f"  page {page_index}: skipped (exists)")
                manifest["pages"].append(
                    {
                        "page": page_index,
                        "status": "skipped",
                        "embedding": embedding_path.name,
                        "image": image_path.name if image_path.exists() else None,
                    }
                )
                continue

            if self._pdf_configuration.save_intermediate_images:
                image.save(
                    str(image_path),
                    self._pdf_configuration.image_format,
                )

            embed_start = time.time()
            try:
                embedding = self._embedder.embed_image(image)
                self._save_embedding(embedding_path, embedding, page_index)
            except Exception as exc:
                embed_seconds = time.time() - embed_start
                print(
                    f"  page {page_index}: FAILED after "
                    f"{embed_seconds:.1f}s: {exc}"
                )
                manifest["pages"].append(
                    {
                        "page": page_index,
                        "status": "error",
                        "error": str(exc),
                        "seconds": embed_seconds,
                    }
                )
                continue
            embed_seconds = time.time() - embed_start

            shape = list(embedding.shape)
            print(
                f"  page {page_index}: embedded in {embed_seconds:.1f}s "
                f"{shape} -> {embedding_path.name}"
            )
            manifest["pages"].append(
                {
                    "page": page_index,
                    "status": "ok",
                    "seconds": embed_seconds,
                    "embedding": embedding_path.name,
                    "embedding_shape": shape,
                    "image": image_path.name
                    if self._pdf_configuration.save_intermediate_images
                    else None,
                }
            )

        (pdf_output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2)
        )

    @staticmethod
    def _save_embedding(
        embedding_path: Path,
        embedding,
        page_index: int,
    ) -> None:
        import torch
        from safetensors.torch import save_file

        save_file(
            {
                "embedding": embedding.detach().cpu().contiguous(),
                "page_index": torch.tensor([page_index], dtype=torch.int64),
            },
            str(embedding_path),
        )
