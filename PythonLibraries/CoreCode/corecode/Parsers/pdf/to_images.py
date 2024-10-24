from pathlib import Path
from pdf2image import convert_from_path
import multiprocessing
from typing import List, Tuple

def process_pdf(
        pdf_file,
        input_dir: Path,
        output_dir: Path,
        image_format: str = 'PNG') -> None:
    """
    Process a single PDF file and convert its pages to images.

    Args:
        pdf_file (Path): The name of the PDF file to process.
        input_dir (Path): The directory containing the PDF file.
        output_dir (Path): The directory where the output images will be saved.
        image_format (str, optional): The format to save the images in. Defaults
        to 'PNG'.

    Returns:
        None
    """
    if isinstance(pdf_file, str):
        pdf_file = Path(pdf_file)
    pdf_path = input_dir / pdf_file
    current_output_dir = output_dir / pdf_file.stem
    current_output_dir.mkdir(parents=True, exist_ok=True)
    
    images = convert_from_path(str(pdf_path))
    for i, image in enumerate(images, start=1):
        image_path = current_output_dir / f'page_{i}.{image_format.lower()}'
        image.save(str(image_path), image_format)
    
    print(f"Processed: {pdf_file}")


def pdf_to_image_parallel(
        input_dir: Path,
        output_dir: Path,
        image_format: str = 'PNG') -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = list(input_dir.glob('*.pdf'))

    args = [(
        pdf_file.name,
        input_dir,
        output_dir,
        image_format) for pdf_file in pdf_files]

    num_processes = multiprocessing.cpu_count()

    with multiprocessing.Pool(processes=num_processes) as pool:
        # Use starmap to pass the arguments as a tuple to the function
        pool.starmap(process_pdf, args)


def convert_pdfs_to_images(
        input_dir: Path,
        output_dir: Path,
        image_format: str = 'PNG') -> None:
    if not input_dir.is_dir():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    
    pdf_to_image_parallel(input_dir, output_dir, image_format)
