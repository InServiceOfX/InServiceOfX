from pathlib import Path
import fal_client

def upload_image_to_fal(base_directory, filename: str):
    """
    Upload an image to fal.ai and get its URL.
    
    Args:
        base_directory: Base directory containing the image
        filename: Name of the image file
        
    Returns:
        str: URL of the uploaded image
        
    Raises:
        FileNotFoundError: If the image file doesn't exist at the specified path
    """
    if isinstance(base_directory, str):
        base_directory = Path(base_directory)
    image_path = base_directory / filename
    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found at {image_path}")
            
    return fal_client.upload_file(str(image_path))