from pathlib import Path
from typing import Union
from lumaai import LumaAI

def upload_image(
    client: LumaAI,
    image_path: Union[Path, str]) -> str:
    """
    Upload an image to Luma AI's CDN
    
    Args:
        client: LumaAI client instance
        image_path: Path to the image file
    
    Returns:
        str: CDN URL of the uploaded image
    """
    with open(str(image_path), 'rb') as f:
        response = client.assets.create(file=f)
    return response.url 