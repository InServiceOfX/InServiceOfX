from pathlib import Path
from typing import List, Optional
# PyMuPDF
import pymupdf
from warnings import warn

class DocumentTextExtraction:
    """Extract text from PDF files."""

    @staticmethod
    def _check_for_valid_file(pdf_path: str | Path) -> tuple[bool, str]:
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            warn(f"PDF file does not exist: {pdf_path}")
            return False, "file does not exist"

        if not pdf_path.suffix.lower() == '.pdf':
            warn(f"File is not a PDF: {pdf_path}")
            return False, "file is not a PDF"

        return True, "valid, existing file"

    @staticmethod
    def extract_text_with_pymupdf(pdf_path: str | Path) -> Optional[str]:
        is_valid, reason = DocumentTextExtraction._check_for_valid_file(
            pdf_path)

        if not is_valid:
            warn(f"Failed to extract text from PDF: {pdf_path} - {reason}")
            return None

        try:
            doc = pymupdf.open(str(pdf_path))
            text = ""
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text += page.get_text()
            doc.close()
            return text.strip()
        except Exception as err:
            warn(f"PyMuPDF extraction failed: {err}")
            return None

    # TODO: Consider not using this, and use pymupdf instead.        
    # @staticmethod
    # def extract_text_with_pypdf2(pdf_path: str | Path) -> Optional[str]:
    #     is_valid, reason = DocumentTextExtraction._check_for_valid_file(
    #         pdf_path)

    #     if not is_valid:
    #         warn(f"Failed to extract text from PDF: {pdf_path} - {reason}")
    #         return None

    #     try:
    #         with open(pdf_path, 'rb') as file:
    #             reader = PyPDF2.PdfReader(file)
    #             text = ""
    #             for page in reader.pages:
    #                 text += page.extract_text()
    #             return text.strip()
    #     except Exception as err:
    #         warn(f"PyPDF2 extraction failed: {err}")
    #         return None

    @staticmethod
    def extract_text_by_pages(pdf_path: str | Path) -> Optional[List[str]]:
        is_valid, reason = DocumentTextExtraction._check_for_valid_file(
            pdf_path)

        if not is_valid:
            warn(f"Failed to extract text from PDF: {pdf_path} - {reason}")
            return None

        try:
            doc = pymupdf.open(str(pdf_path))
            page_texts = []
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text().strip()
                if text:  # Only add non-empty pages
                    page_texts.append(text)
            doc.close()
            return page_texts
        except Exception as err:
            warn(f"Page-by-page extraction failed: {err}")
            return None