from pathlib import Path
import pandas as pd
from corecode.Parsers.pdf.parse_tables import parse_tables
# Path to the test PDF file
TEST_PDF_PATH = Path(__file__).parents[2] / "Data" / \
    "nvidia-h100-datasheet-2430615.pdf"

def test_parse_pdf_tables():
    # Ensure the test PDF file exists
    assert TEST_PDF_PATH.exists(), f"Test PDF file not found: {TEST_PDF_PATH}"

    # Parse tables from the PDF
    tables = parse_tables(TEST_PDF_PATH)

    # Check if any tables were extracted
    assert len(tables) > 0, "No tables were extracted from the PDF"
    assert len(tables) == 3

    assert all(isinstance(table, pd.DataFrame) for table in tables), \
        "Not all extracted tables are pandas DataFrames"

    assert tables[1].shape == (36, 4)

def parse_table(df):
    parsed_rows = []
    headers = df.iloc[1].tolist()
    
    for _, row in df.iloc[2:].iterrows():
        if pd.isna(row.iloc[0]):
            continue
        
        parsed_row = {"Description": row.iloc[0]}
        
        for i, (header, value) in enumerate(
            zip(headers[1:], row.iloc[1:]), start=1):
            if not pd.isna(header):
                parsed_row[header] = value if not pd.isna(value) else None
        
        parsed_rows.append(parsed_row)
    
    return parsed_rows

def test_parse_nvidia_datasheet():
    tables = parse_tables(TEST_PDF_PATH)
    rows = parse_table(tables[1])

    assert len(rows) == 17
    assert len(rows[0]) == 4
    assert rows[0]["Description"] == "FP64 Tensor Core"
    assert rows[0]["34 teraFLOPS"] == "67 teraFLOPS"