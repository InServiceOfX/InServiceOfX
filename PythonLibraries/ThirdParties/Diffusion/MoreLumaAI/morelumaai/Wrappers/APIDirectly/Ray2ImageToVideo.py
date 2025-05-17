import requests
import time
from pathlib import Path
from typing import Optional, Dict, Any

from morelumaai.Configuration.APIDirectlyConfigurations import Ray2Configuration

class Ray2ImageToVideo:
    # API endpoints as class constants
    API_ENDPOINT_GENERATIONS = "/dream-machine/v1/generations"
    
    def __init__(self, base_url: str = "https://api.lumalabs.ai"):
        self.base_url = base_url
        self.current_generation = None
        self.current_video_url = None
    
    def _create_headers(self, api_key: str) -> Dict:
        return {
            "accept": "application/json",
            "authorization": f"Bearer {api_key}",
            "content-type": "application/json"
        }
    
    def create_payload(
            self,
            prompt: str,
            image_url: str,
            config = None) -> Dict:
        """
        Create the payload for Ray 2 Image to Video generation
        
        Args:
            prompt: Text description of the desired video
            image_url: URL of the starting image
            config: Optional Ray2Configuration object
            
        Returns:
            Dict: API payload dictionary
        """
        if config is None:
            config = Ray2Configuration()
            
        return {
            "prompt": prompt,
            **config.to_api_kwargs(),
            "keyframes": {
                "frame0": {
                    "type": "image",
                    "url": image_url
                }
            }
        }
    
    def generate(
            self, api_key: str,
            prompt: str,
            image_url: str,
            config = None) -> Dict[str, Any]:
        """
        Generate video from image using Ray 2 model
        
        Args:
            api_key: LumaAI API key
            prompt: Text description of the desired video
            image_url: URL of the starting image
            config: Optional Ray2Configuration object
            
        Returns:
            Dict: Complete generation response object
        """
        # Get headers with the API key
        headers = self._create_headers(api_key)
        
        # Create the payload using the dedicated method
        payload = self.create_payload(prompt, image_url, config)
        
        # Make API request
        try:
            response = requests.post(
                f"{self.base_url}{self.API_ENDPOINT_GENERATIONS}", 
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            generation = response.json()
            
            # Monitor progress
            generation_id = generation["id"]
            start_time = time.time()
            completed = False
            
            while not completed:
                status_response = requests.get(
                    f"{self.base_url}{self.API_ENDPOINT_GENERATIONS}/{generation_id}",
                    headers=headers
                )
                status_response.raise_for_status()
                generation = status_response.json()
                
                if generation["state"] == "completed":
                    completed = True
                    self.current_video_url = generation["assets"]["video"]
                elif generation["state"] == "failed":
                    print(f"Generation failed: {generation.get('failure_reason', 'Unknown error')}")
                    self.current_video_url = None
                    completed = True
                    
                elapsed_time = time.time() - start_time
                print(
                    f"State: {generation['state']} - Time elapsed: {elapsed_time:.2f}s")
                time.sleep(3)
                
            total_time = time.time() - start_time
            print(f"Total generation time: {total_time:.2f}s")
            
            # Always store the generation response
            self.current_generation = generation
            
            # Return the complete generation response
            return generation
            
        except requests.exceptions.RequestException as e:
            error_response = {
                "state": "error",
                "error": str(e),
                "status_code": getattr(e.response, "status_code", None) if hasattr(e, "response") else None
            }
            self.current_generation = error_response
            self.current_video_url = None
            return error_response
    
    def save_video(self, api_key: str = None, save_path: Optional[Path] = None) -> Path:
        """
        Save the generated video to disk
        
        Args:
            api_key: Optional API key for authentication if needed for video download
            save_path: Optional custom path to save the video
            
        Returns:
            Path: Path where the video was saved
        """
        if not self.current_video_url:
            raise RuntimeError(
                "No video has been generated yet or generation failed. Check current_generation for details.")
            
        if save_path is None:
            save_path = Path(
                f"ray2_generation_{self.current_generation['id']}.mp4")
            
        # For video download, we may not need authentication, but it's good to have the option
        headers = None
        if api_key:
            headers = self._create_headers(api_key)
            
        response = requests.get(self.current_video_url, headers=headers, stream=True)
        response.raise_for_status()
        
        with open(str(save_path), 'wb') as file:
            file.write(response.content)
        print(f"File downloaded as {save_path}")
        
        return save_path
