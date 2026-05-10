from pathlib import Path
from typing import Iterator, Tuple

from PIL import Image
from pdf2image import convert_from_path


class PDFRasterizer:
    def __init__(self, dpi: int = 250):
        self._dpi = dpi

    def rasterize(self, pdf_path: Path) -> list[Image.Image]:
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        return convert_from_path(str(pdf_path), dpi=self._dpi)

    def iter_pages(
        self,
        pdf_path: Path,
    ) -> Iterator[Tuple[int, Image.Image]]:
        """Yield (page_number, image) starting at page 1.

        Loads all pages eagerly via pdf2image (it returns a list); this is the
        same behaviour as corecode.Parsers.pdf.to_images. For very large PDFs
        we'd want to switch to per-page rendering with pdftoppm directly.
        """
        for index, image in enumerate(self.rasterize(pdf_path), start=1):
            yield index, image
