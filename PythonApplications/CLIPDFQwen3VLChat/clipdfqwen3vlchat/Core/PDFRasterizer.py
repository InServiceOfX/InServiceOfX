from pathlib import Path
from typing import Iterator, Tuple

from PIL import Image
from pdf2image import convert_from_path


class PDFRasterizer:
    """Mirror of CLIPDFExtraction's PDFRasterizer.

    Kept as a local copy rather than imported across sister apps so the two
    apps don't develop a hard dependency on each other's package layout.
    """

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
        for index, image in enumerate(self.rasterize(pdf_path), start=1):
            yield index, image
