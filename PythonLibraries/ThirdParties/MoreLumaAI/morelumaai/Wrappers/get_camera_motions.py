from typing import List
import requests

def get_camera_motions(api_key: str):
    """
    Get list of available camera motion presets from LumaAI
    
    Args:
        api_key: Optional API key. If not provided, uses value from .env.
    client
        
    Returns:
        List[CameraMotion]: List of available camera motion presets
        
    Raises:
        RuntimeError: If the API request fails
    """    
    url = "https://api.lumalabs.ai/dream-machine/v1/generations/camera_motion/list"
    headers = {
        'accept': 'application/json',
        'authorization': f'Bearer {api_key}'
    }
    
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        raise RuntimeError(
            f"Failed to get camera motions: {response.status_code} - {response.text}")
    
    return response.json()