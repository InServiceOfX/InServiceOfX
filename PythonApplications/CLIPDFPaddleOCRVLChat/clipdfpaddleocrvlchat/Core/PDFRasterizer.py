from pathlib import Path
from typing import Iterator, Tuple

from pdf2image import convert_from_path
from PIL import Image


class PDFRasterizer:
    def __init__(self, dpi: int = 250):
        self._dpi = dpi

    def iter_pages(self, pdf_path: Path) -> Iterator[Tuple[int, Image.Image]]:
        pages = convert_from_path(str(pdf_path), dpi=self._dpi)
        for page_index, image in enumerate(pages, start=1):
            yield page_index, image.convert("RGB")
