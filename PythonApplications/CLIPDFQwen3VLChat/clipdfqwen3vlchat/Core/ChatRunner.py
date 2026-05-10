import json
import time
from pathlib import Path

from clipdfqwen3vlchat.Core.PDFChatConfiguration import PDFChatConfiguration
from clipdfqwen3vlchat.Core.PDFRasterizer import PDFRasterizer


class ChatRunner:
    """Per-PDF, per-page Qwen3-VL chat driver.

    Mirrors CLIPDFExtraction's ExtractionRunner but writes freeform .txt per
    page (the Qwen3-VL response) instead of structured .md per page.
    """

    def __init__(
        self,
        qwen3vl_runner,
        pdf_configuration: PDFChatConfiguration,
    ):
        self._qwen3vl_runner = qwen3vl_runner
        self._pdf_configuration = pdf_configuration
        self._rasterizer = PDFRasterizer(dpi=pdf_configuration.pdf_dpi)

    def run(self) -> None:
        if not self._qwen3vl_runner.is_loaded():
            print("Loading Qwen3-VL model into vLLM...")
            self._qwen3vl_runner.load()
            print("Qwen3-VL loaded.")

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
            "prompt": self._pdf_configuration.prompt,
            "pages": [],
        }

        for page_index, image in pages:
            txt_path = pdf_output_dir / f"page_{page_index}.txt"
            png_path = (
                pdf_output_dir
                / f"page_{page_index}.{self._pdf_configuration.image_format.lower()}"
            )

            if (
                self._pdf_configuration.skip_existing
                and txt_path.exists()
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

            chat_start = time.time()
            try:
                response = self._qwen3vl_runner.generate(
                    image,
                    self._pdf_configuration.prompt,
                )
            except Exception as exc:
                chat_seconds = time.time() - chat_start
                print(
                    f"  page {page_index}: FAILED after "
                    f"{chat_seconds:.1f}s: {exc}"
                )
                manifest["pages"].append(
                    {
                        "page": page_index,
                        "status": "error",
                        "error": str(exc),
                        "seconds": chat_seconds,
                    }
                )
                continue
            chat_seconds = time.time() - chat_start

            txt_path.write_text(response)
            print(
                f"  page {page_index}: answered in {chat_seconds:.1f}s "
                f"-> {txt_path.name}"
            )
            manifest["pages"].append(
                {
                    "page": page_index,
                    "status": "ok",
                    "seconds": chat_seconds,
                    "output": txt_path.name,
                }
            )

        (pdf_output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2)
        )
