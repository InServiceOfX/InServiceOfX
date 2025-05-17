from pathlib import Path
from typing import Optional, Dict
from lumaai import LumaAI
from morelumaai.Configuration import GenerationConfiguration
import requests
import time

class GenerateVideo:
    def __init__(
            self,
            configuration: GenerationConfiguration,
            api_key: str = None):
        self.configuration = configuration
        if api_key is None:
            self.client = LumaAI()
        else:
            self.client = LumaAI(auth_token=api_key)

    def generate(self, prompt: str, keyframes: Optional[Dict] = None):
        """
        Generate a video using LumaAI
        
        Args:
            prompt: Text description of the desired video
            keyframes: Optional dictionary for image-based generation

        Returns:
            str: URL of the generated video
            
        Raises:
            RuntimeError: If generation fails
        """
        # Prepare generation parameters
        generation_params = {
            "prompt": prompt,
            **self.configuration.to_api_kwargs()
        }
        if keyframes:
            generation_params["keyframes"] = keyframes

        # Start generation
        generation = self.client.generations.create(**generation_params)
        
        # Monitor progress
        start_time = time.time()
        completed = False
        while not completed:
            generation = self.client.generations.get(id=generation.id)
            if generation.state == "completed":
                completed = True
            elif generation.state == "failed":
                raise RuntimeError(
                    f"Generation failed: {generation.failure_reason}")
            elapsed_time = time.time() - start_time
            print(
                f"State: {generation.state} - Time elapsed: {elapsed_time:.2f}s")
            time.sleep(3)

        total_time = time.time() - start_time
        print(f"Total generation time: {total_time:.2f}s")

        self.current_generation = generation
        self.current_video_url = generation.assets.video

        return self.current_video_url

    def save_video(self) -> Path:
        """
        Save the generated video to disk using the configuration's
        temporary_save_path
        
        Returns:
            Path: Path where the video was saved
            
        Raises:
            RuntimeError: If no video has been generated yet
        """
        if not self.current_generation or not self.current_video_url:
            raise RuntimeError(
                "No video has been generated yet. Call generate() first.")

        save_path = Path(self.configuration.temporary_save_path) / \
            f"{self.current_generation.id}.mp4"

        response = requests.get(self.current_video_url, stream=True)
        with open(str(save_path), 'wb') as file:
            file.write(response.content)
        print(f"File downloaded as {save_path}")
        
        return save_path

    # https://docs.lumalabs.ai/docs/video-generation#with-start-and-end-keyframes

    @staticmethod
    def create_start_keyframe(image_url: str) -> Dict:
        """
        Create a starting keyframe dictionary for video generation
        
        Args:
            image_url: URL of the image to use as starting frame
            
        Returns:
            Dict: Keyframe dictionary with frame0 configuration
        """
        return {
            "frame0": {
                "type": "image",
                "url": image_url
            }
        }

    @staticmethod
    def create_end_keyframe(image_url: str) -> Dict:
        """
        Create an ending keyframe dictionary for video generation
        
        Args:
            image_url: URL of the image to use as ending frame
            
        Returns:
            Dict: Keyframe dictionary with frame1 configuration
        """
        return {
            "frame1": {
                "type": "image",
                "url": image_url
            }
        }

    def set_loop(self, loop: bool = False):
        if loop is False:
            self.configuration.loop = None
        else:
            self.configuration.loop = True

    def list_all_generations(self, limit: int = 100):
        self.current_generations_list = self.client.generations.list(
            limit=limit,
            offset=0)
        return self.current_generations_list

    def delete_generation(self, generation_id: str) -> None:
        self.client.generations.delete(id=generation_id)