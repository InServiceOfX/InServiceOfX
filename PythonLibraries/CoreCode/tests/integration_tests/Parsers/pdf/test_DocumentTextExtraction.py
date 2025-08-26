from corecode.Parsers.pdf.DocumentTextExtraction import DocumentTextExtraction

from pathlib import Path
import pytest

test_data_path = Path(__file__).parents[3] / "TestData"

is_RCC_319_pdf_exists = (test_data_path / "RCC_319.pdf").exists()

skip_RCC_319_test_reason = "RCC_319.pdf does not exist"

@pytest.mark.skipif(not is_RCC_319_pdf_exists, reason=skip_RCC_319_test_reason)
def test_extract_RCC_319_with_DocumentTextExtraction():
    text = DocumentTextExtraction.extract_text_with_pymupdf(
        test_data_path / "RCC_319.pdf")
    assert text is not None
    assert len(text) > 0
    assert "RCC 319" in text