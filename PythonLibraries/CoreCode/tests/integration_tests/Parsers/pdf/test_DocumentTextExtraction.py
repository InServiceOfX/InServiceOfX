from corecode.Parsers.pdf.DocumentTextExtraction import DocumentTextExtraction

from pathlib import Path
import pytest

test_data_path = Path(__file__).parents[3] / "TestData"

# https://apps.dtic.mil/sti/citations/AD1075829
is_AD1075829_pdf_exists = (test_data_path / "AD1075829.pdf").exists()

skip_AD1075829_test_reason = "AD1075829.pdf does not exist"

@pytest.mark.skipif(
        not is_AD1075829_pdf_exists,
        reason=skip_AD1075829_test_reason)
def test_extract_AD1075829_with_DocumentTextExtraction():
    text = DocumentTextExtraction.extract_text_with_pymupdf(
        test_data_path / "AD1075829.pdf")
    assert text is not None
    assert len(text) > 0

    # print(text[:500])
    # 1350531
    # print(len(text))

    text_pages = DocumentTextExtraction.extract_text_by_pages(
        test_data_path / "AD1075829.pdf")
    assert text_pages is not None
    assert len(text_pages) > 0

    for page_num, page_text in enumerate(text_pages):
        if page_num > 7:
            break
        print(f"Page {page_num}:")
        print(page_text[:500])
        print(f"Length: {len(page_text)}")


