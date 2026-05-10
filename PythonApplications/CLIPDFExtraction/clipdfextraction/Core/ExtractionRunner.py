import json
import time
from pathlib import Path

from clipdfextraction.Core.PDFExtractionConfiguration import (
    PDFExtractionConfiguration,
)
from clipdfextraction.Core.PDFRasterizer import PDFRasterizer


class ExtractionRunner:
    def __init__(
        self,
        mineru_runner,
        pdf_configuration: PDFExtractionConfiguration,
    ):
        self._mineru_runner = mineru_runner
        self._pdf_configuration = pdf_configuration
        self._rasterizer = PDFRasterizer(dpi=pdf_configuration.pdf_dpi)

    def run(self) -> None:
        if not self._mineru_runner.is_loaded():
            print("Loading MinerU model into vLLM...")
            self._mineru_runner.load()
            print("MinerU loaded.")

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
        pdf_output_dir = (
            self._pdf_configuration.output_path / pdf_path.stem
        )
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
            "num_pages": len(pages),
            "dpi": self._pdf_configuration.pdf_dpi,
            "pages": [],
        }

        for page_index, image in pages:
            md_path = pdf_output_dir / f"page_{page_index}.md"
            png_path = (
                pdf_output_dir
                / f"page_{page_index}.{self._pdf_configuration.image_format.lower()}"
            )

            if (
                self._pdf_configuration.skip_existing
                and md_path.exists()
            ):
                print(f"  page {page_index}: skipped (exists)")
                manifest["pages"].append(
                    {"page": page_index, "status": "skipped"}
                )
                continue

            if self._pdf_configuration.save_intermediate_images:
                image.save(
                    str(png_path),
                    self._pdf_configuration.image_format,
                )

            extract_start = time.time()
            try:
                result = self._mineru_runner.extract_from_image(image)
            except Exception as exc:
                extract_seconds = time.time() - extract_start
                print(
                    f"  page {page_index}: FAILED after "
                    f"{extract_seconds:.1f}s: {exc}"
                )
                manifest["pages"].append(
                    {
                        "page": page_index,
                        "status": "error",
                        "error": str(exc),
                        "seconds": extract_seconds,
                    }
                )
                continue
            extract_seconds = time.time() - extract_start

            md_path.write_text(self._stringify_result(result))
            print(
                f"  page {page_index}: extracted in {extract_seconds:.1f}s "
                f"-> {md_path.name}"
            )
            manifest["pages"].append(
                {
                    "page": page_index,
                    "status": "ok",
                    "seconds": extract_seconds,
                    "output": md_path.name,
                }
            )

        (pdf_output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2)
        )

    @staticmethod
    def _stringify_result(result) -> str:
        # MinerUClient.two_step_extract returns a structure (list of blocks /
        # markdown / etc. depending on the model card). Stringify defensively
        # so we always write *something* even if the shape changes.
        if isinstance(result, str):
            return result
        try:
            return json.dumps(result, indent=2, default=str)
        except (TypeError, ValueError):
            return repr(result)
