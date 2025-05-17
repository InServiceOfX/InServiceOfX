import time
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
from lumaai import LumaAI
import requests

class CreateImage:
    """Class for generating images using LumaAI's Python SDK"""
    
    def __init__(self, api_key: str = None):
        """
        Initialize the CreateImage client
        
        Args:
            api_key: Optional LumaAI API key (if not provided, uses environment variable)
        """
        if api_key is None:
            self.client = LumaAI()
        else:
            self.client = LumaAI(auth_token=api_key)
            
        self.current_generation = None
        self.current_image_url = None
    
    @staticmethod
    def create_image_ref(url: str, weight: float = 0.85) -> List[Dict]:
        """
        Create an image reference list for use with generate
        
        Args:
            url: URL of the reference image
            weight: How strongly to weight the reference image (0.0-1.0)
            
        Returns:
            List[Dict]: Image reference object in the format expected by the API
        """
        return [{
            "url": url,
            "weight": weight
        }]
    
    @staticmethod
    def create_modify_image_ref(url: str, weight: float = None) -> Dict:
        """
        Create a modify image reference object for use with generate
        
        Args:
            url: URL of the image to modify
            
        Returns:
            Dict: Modify image reference object in the format expected by the API
        """
        if weight is None:
            return {
                "url": url
            }
        else:
            return {
                "url": url,
                "weight": weight
            }
    
    @staticmethod
    def create_style_ref(url: str, weight: float = 0.85) -> List[Dict]:
        """
        Create a style reference list for use with generate
        
        Args:
            url: URL of the style reference image
            weight: How strongly to weight the style reference (0.0-1.0)
            
        Returns:
            List[Dict]: Style reference object in the format expected by the API
        """
        return [{
            "url": url,
            "weight": weight
        }]
    
    def generate(
            self,
            prompt: str,
            model: str = "photon-1",
            aspect_ratio: Optional[str] = None,
            image_ref: Optional[List[Dict]] = None,
            modify_image_ref: Optional[Dict] = None,
            style_ref: Optional[List[Dict]] = None,
            callback_url: Optional[str] = None) -> Any:
        """
        Returns:
            Any: Complete generation response object
        """
        # Validate model
        if model not in ["photon-1", "photon-flash-1"]:
            raise ValueError(f"Model must be one of photon-1 or photon-flash-1, got {model}")
        
        # Prepare generation parameters
        generation_params = {
            "prompt": prompt,
            "model": model,
        }
        
        # Add optional parameters
        if aspect_ratio:
            generation_params["aspect_ratio"] = aspect_ratio
            
        if image_ref:
            generation_params["image_ref"] = image_ref
            
        if modify_image_ref:
            generation_params["modify_image_ref"] = modify_image_ref
            
        if style_ref:
            generation_params["style_ref"] = style_ref
            
        if callback_url:
            generation_params["callback_url"] = callback_url
        
        try:
            # Start generation
            generation = self.client.generations.image.create(
                **generation_params)
            
            # Monitor progress
            start_time = time.time()
            completed = False
            
            while not completed:
                generation = self.client.generations.get(id=generation.id)
                
                if generation.state == "completed":
                    completed = True
                    self.current_image_url = generation.assets.image
                elif generation.state == "failed":
                    print(f"Generation failed: {generation.failure_reason}")
                    print(f"Generation params: {generation_params}")
                    self.current_image_url = None
                    completed = True
                
                elapsed_time = time.time() - start_time
                print(f"State: {generation.state} - Time elapsed: {elapsed_time:.2f}s")
                time.sleep(3)
            
            total_time = time.time() - start_time
            print(f"Total generation time: {total_time:.2f}s")
            
            # Store the generation
            self.current_generation = generation
            
            # Return the complete generation object
            return generation
            
        except Exception as e:
            print(f"Error during image generation: {str(e)}")
            self.current_generation = None
            self.current_image_url = None
            raise
    
    def save_image(self, save_path: Optional[Path] = None) -> Path:
        """
        Args:
            save_path: Optional custom path to save the image
            
        Returns:
            Path: Path where the image was saved
        """
        if not self.current_generation or not self.current_image_url:
            raise RuntimeError(
                "No image has been generated yet or generation failed. Call generate() first.")
            
        if save_path is None:
            save_path = Path(f"image_generation_{self.current_generation.id}.png")
            
        response = requests.get(self.current_image_url, stream=True)
        response.raise_for_status()
        
        with open(str(save_path), 'wb') as file:
            file.write(response.content)
        print(f"File downloaded as {save_path}")
        
        return save_path
