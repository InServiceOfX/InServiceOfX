from typing import List, Dict, Union, Optional

def create_vision_message(
    text: str,
    image_urls: Union[str, List[str]],
    role: str = "user"
) -> Dict:
    """
    Create a message with both text and image content for vision models.
    
    Args:
        text: The text prompt to accompany the image(s)
        image_urls: Single URL string or list of URL strings pointing to images
        role: The role of the message sender (default: "user")
        
    Returns:
        Dict: A properly formatted message for vision models
    """
    # Convert single URL to list for consistent handling
    if isinstance(image_urls, str):
        image_urls = [image_urls]
    
    # Start with the text content
    content = [
        {
            "type": "text",
            "text": text
        }
    ]
    
    # Add each image URL to the content
    for url in image_urls:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": url
            }
        })
    
    return {
        "role": role,
        "content": content
    }
