import pandas as pd
import tabula
from pathlib import Path

def parse_tables(pdf_path):
    """
    Parse tables from a PDF file and return them as pandas DataFrames.

    Args:
    pdf_path (str or Path): Path to the PDF file.

    Returns:
    list: A list of pandas DataFrames, each representing a table from the PDF.
    """
    # Convert to Path object if it's a string
    pdf_path = Path(pdf_path)

    # Resolve the path (handle relative paths and get absolute path)
    pdf_path = pdf_path.resolve()

    # Check if the file exists
    if not pdf_path.exists():
        raise FileNotFoundError(f"The PDF file does not exist: {pdf_path}")

    # Parse tables from the PDF
    tables = tabula.read_pdf(str(pdf_path), pages='all', multiple_tables=True)

    # Filter out empty DataFrames
    tables = [table for table in tables if not table.empty]

    return tables