from pathlib import Path
import shutil
import pytest
import sys
import subprocess

# Check if pdf2image is installed, if not, install it
try:
    import pdf2image
except (ModuleNotFoundError, ImportError):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pdf2image"])
    import pdf2image

from corecode.Parsers.pdf.to_images import (
    convert_pdfs_to_images,
    process_pdf)
from PIL import Image

# Path to the test PDF files
test_data_directory = Path(__file__).parents[2] / "Data"
test_nvidia_pdf_path = test_data_directory / "nvidia-h100-datasheet-2430615.pdf"
test_output_directory = Path(__file__).parents[2] / "Data" / "Output" / "PDFImages"

# def setup_module(module):
#     """Ensure the test output directory exists and is empty."""
#     if test_output_directory.exists():
#         shutil.rmtree(test_output_directory)
#     test_output_directory.mkdir(parents=True)

# def teardown_module(module):
#     """Clean up the test output directory after all tests."""
#     if test_output_directory.exists():
#         shutil.rmtree(test_output_directory)

# def test_convert_pdfs_to_images_png():
#     # Ensure the test data directory exists
#     assert test_data_directory.is_dir(), f"Test data directory does not exist: {test_data_directory}"

#     # Convert PDFs to PNG images
#     convert_pdfs_to_images(test_data_directory, test_output_directory)

#     # Check if output directory was created
#     assert test_output_directory.is_dir(), f"Output directory was not created: {test_output_directory}"

#     # Check if images were created for each PDF
#     for pdf_file in test_data_directory.glob('*.pdf'):
#         pdf_output_dir = test_output_directory / pdf_file.stem
#         assert pdf_output_dir.is_dir(), f"Output directory for {pdf_file.name} was not created"

#         # Check if at least one PNG image was created
#         png_files = list(pdf_output_dir.glob('*.png'))
#         assert len(png_files) > 0, f"No PNG images were created for {pdf_file.name}"

# def test_convert_pdfs_to_images_jpeg():
#     # Convert PDFs to JPEG images
#     convert_pdfs_to_images(test_data_directory, test_output_directory, image_format='JPEG')

#     # Check if JPEG images were created for each PDF
#     for pdf_file in test_data_directory.glob('*.pdf'):
#         pdf_output_dir = test_output_directory / pdf_file.stem
#         assert pdf_output_dir.is_dir(), f"Output directory for {pdf_file.name} was not created"

#         # Check if at least one JPEG image was created
#         jpeg_files = list(pdf_output_dir.glob('*.jpeg'))
#         assert len(jpeg_files) > 0, f"No JPEG images were created for {pdf_file.name}"

# def test_convert_pdfs_to_images_invalid_input():
#     invalid_input_dir = Path("/nonexistent/directory")
#     with pytest.raises(ValueError) as excinfo:
#         convert_pdfs_to_images(invalid_input_dir, test_output_directory)
#     assert str(excinfo.value) == f"Input directory does not exist: {invalid_input_dir}"

# def test_convert_pdfs_to_images_empty_directory():
#     empty_dir = test_output_directory / "empty"
#     empty_dir.mkdir(parents=True, exist_ok=True)

#     convert_pdfs_to_images(empty_dir, test_output_directory)

#     # Check that no output was created for an empty input directory
#     assert len(list(test_output_directory.glob('*'))) == 1, "Output was created for an empty input directory"

def test_process_pdf():
    # Setup test directories
    test_data_directory = Path(__file__).parents[2] / "Data"
    test_pdf_path = test_data_directory / "nvidia-h100-datasheet-2430615.pdf"
    test_output_directory = Path(__file__).parents[2] / "Data" / "Output" / "PDFImages"
    
    # Ensure the test PDF file exists
    assert test_pdf_path.exists(), f"Test PDF file not found: {test_pdf_path}"
    
    # Create output directory if it doesn't exist
    test_output_directory.mkdir(parents=True, exist_ok=True)
    
    # Process the PDF
    process_pdf(test_pdf_path.name, test_data_directory, test_output_directory)
    
    # Check if output directory for this PDF was created
    pdf_output_dir = test_output_directory / test_pdf_path.stem
    assert pdf_output_dir.is_dir(), f"Output directory for {test_pdf_path.name} was not created"
    
    # Check if PNG images were created
    png_files = list(pdf_output_dir.glob('*.png'))
    assert len(png_files) > 0, f"No PNG images were created for {test_pdf_path.name}"
    
    # Check if the first image is a valid PNG
    first_image_path = png_files[0]
    with Image.open(first_image_path) as img:
        assert img.format == 'PNG', f"First image is not in PNG format: {first_image_path}"
    
    # Test with JPEG format
    process_pdf(test_pdf_path.name, test_data_directory, test_output_directory, image_format='JPEG')
    
    # Check if JPEG images were created
    jpeg_files = list(pdf_output_dir.glob('*.jpeg'))
    assert len(jpeg_files) > 0, f"No JPEG images were created for {test_pdf_path.name}"
    
    # Check if the first image is a valid JPEG
    first_jpeg_path = jpeg_files[0]
    with Image.open(first_jpeg_path) as img:
        assert img.format == 'JPEG', f"First image is not in JPEG format: {first_jpeg_path}"
