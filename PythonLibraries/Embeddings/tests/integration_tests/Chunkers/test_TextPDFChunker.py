from corecode.Parsers.pdf.DocumentTextExtraction import DocumentTextExtraction
from corecode.Utilities import DataSubdirectories, is_model_there

from embeddings.Chunkers import TextPDFChunker

from pathlib import Path
import pytest

core_code_test_data_path = Path(__file__).parents[4] / "CoreCode" / "tests" / \
    "TestData"

is_AD1075829_pdf_exists = (core_code_test_data_path / "AD1075829.pdf").exists()

skip_AD1075829_test_reason = "AD1075829.pdf does not exist"

data_subdirectories = DataSubdirectories()
relative_model_path = "Models/Embeddings/BAAI/bge-large-en-v1.5"

is_model_downloaded, model_path = is_model_there(
    relative_model_path,
    data_subdirectories)

model_is_not_downloaded_message = f"Model {relative_model_path} not downloaded"

# TODO: I obtained this error:
# Chunkers/test_TextPDFChunker.py Token indices sequence length is longer than the specified maximum sequence length for this model (4239 > 512). Running this sequence through the model will result in indexing errors
@pytest.mark.skipif(
        not is_AD1075829_pdf_exists or not is_model_downloaded,
        reason=skip_AD1075829_test_reason + " or " + \
            model_is_not_downloaded_message)
def test_TextPDFChunker_chunk_page_by_page():
    text_pages = DocumentTextExtraction.extract_text_by_pages(
        core_code_test_data_path / "AD1075829.pdf")

    text_pdf_chunker = TextPDFChunker(model_path)    

    text_pdf_chunks_by_page = text_pdf_chunker.chunk_pdf_by_pages(
        text_pages,
        str(core_code_test_data_path / "AD1075829.pdf"))

    assert text_pdf_chunks_by_page is not None
    assert len(text_pdf_chunks_by_page) > 0

    print(f"Number of chunks by page: {len(text_pdf_chunks_by_page)}")
    for text_pdf_chunk in text_pdf_chunks_by_page[:5]:
        print(text_pdf_chunk.content[:500])
        print(f"Length: {len(text_pdf_chunk.content)}")

# TODO: I obtained this error:
# Chunkers/test_TextPDFChunker.py Token indices sequence length is longer than the specified maximum sequence length for this model (299862 > 512). Running this sequence through the model will result in indexing errors
@pytest.mark.skipif(
        not is_AD1075829_pdf_exists or not is_model_downloaded,
        reason=skip_AD1075829_test_reason + " or " + \
            model_is_not_downloaded_message)
def test_TextPDFChunker_chunk_full_text():
    text_pdf_chunker = TextPDFChunker(model_path)    

    text = DocumentTextExtraction.extract_text_with_pymupdf(
        core_code_test_data_path / "AD1075829.pdf")

    text_pdf_chunks = text_pdf_chunker.chunk_pdf_text(
        text,
        str(core_code_test_data_path / "AD1075829.pdf"))

    assert text_pdf_chunks is not None
    assert len(text_pdf_chunks) > 0

    print(f"Number of chunks: {len(text_pdf_chunks)}")
    for text_pdf_chunk in text_pdf_chunks[:5]:
        print(text_pdf_chunk.content[:500])
        print(f"Length: {len(text_pdf_chunk.content)}")