import requests
from typing import Tuple, Optional
import time

def check_api_endpoint(
    url: str, 
    headers: Optional[dict] = None,
    timeout: int = 5,
    max_retries: int = 3,
    retry_delay: int = 1
) -> Tuple[bool, Optional[str]]:
    """
    Check if an API endpoint exists and is reachable
    
    Args:
        url: The URL to check
        headers: Optional headers to include in the request
        timeout: Request timeout in seconds
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds
        
    Returns:
        Tuple of (is_reachable, error_message)
    """
    attempt = 0
    while attempt < max_retries:
        attempt += 1
        try:
            # Use HEAD request to minimize data transfer
            response = requests.head(url, headers=headers, timeout=timeout)
            
            # Some APIs don't support HEAD requests, fall back to GET
            if response.status_code == 405:  # Method Not Allowed
                response = requests.get(url, headers=headers, timeout=timeout)
            
            # Check for success status codes (2xx) or functioning API codes
            if 200 <= response.status_code < 300 or response.status_code == 401:
                # 401 means the API exists but requires authentication
                return True, None
            else:
                return False, f"Endpoint returned status code: {response.status_code}"
                
        except requests.exceptions.Timeout:
            error = f"Timeout connecting to {url}"
        except requests.exceptions.ConnectionError:
            error = f"Connection error for {url}"
        except requests.exceptions.RequestException as e:
            error = f"Request failed: {str(e)}"
            
        # If we've reached max retries, return failure
        if attempt >= max_retries:
            return False, error
            
        # Otherwise wait and retry
        time.sleep(retry_delay)
    
    return False, "Maximum retries exceeded"
