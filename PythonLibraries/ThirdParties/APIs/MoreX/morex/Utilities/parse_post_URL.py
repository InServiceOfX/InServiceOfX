from urllib.parse import urlparse

def parse_post_URL(url: str) -> tuple[str, str]:
    """
    Parse Twitter/X URL to extract author and post ID.
    
    Args:
        url: Twitter/X URL with guaranteed structure
        
    Returns:
        tuple: (author, post_id)
    """
    parsed = urlparse(url)
    path_parts = parsed.path.strip('/').split('/')
    
    if len(path_parts) >= 3 and path_parts[1] == 'status':
        author = path_parts[0]
        post_id = path_parts[2]
        return author, post_id
    
    raise ValueError(f"Invalid X URL structure: {url}")
