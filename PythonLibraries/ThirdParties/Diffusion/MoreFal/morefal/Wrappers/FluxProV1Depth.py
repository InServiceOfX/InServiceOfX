from typing import Optional
import fal_client
from dataclasses import dataclass
import time
import asyncio

@dataclass
class FluxProV1DepthResult:
    url: str
    width: int
    height: int
    content_type: str
    inference_time: float
    seed: int
    has_nsfw_concepts: bool

class FluxProV1Depth:
    def __init__(self):
        self.application_name = "fal-ai/flux-pro/v1/depth"
        self.safety_tolerance = 6
        self.wait_time = 3  # seconds between event checks

    async def generate(
            self, 
            prompt: str,
            image_url: str,
            guidance_scale: float = 3.5,
            num_inference_steps: int = 28,
            enable_logs: bool = True) -> FluxProV1DepthResult:
        """
        Generate an image using Flux Pro Depth model.
        
        Args:
            prompt: Text description of the desired image transformation
            image_url: URL of the input image
            guidance_scale: Controls how closely to follow the prompt
            (default: 3.5)
            num_inference_steps: Number of steps in the diffusion process
            (default: 28)
            enable_logs: Whether to print generation logs (default: True)
        
        Returns:
            FluxProDepthResult containing the generated image details
        """
        handler = await fal_client.submit_async(
            self.application_name,
            arguments={
                "control_image_url": image_url,
                "prompt": prompt,
                "safety_tolerance": self.safety_tolerance,
                "guidance_scale": guidance_scale,
                "num_inference_steps": num_inference_steps
            },
        )
        
        start_time = time.time()
        if enable_logs:
            async for event in handler.iter_events(with_logs=True):
                elapsed_time = time.time() - start_time
                print(f"Event: {event} - Time elapsed: {elapsed_time:.2f}s")
                await asyncio.sleep(self.wait_time)

        result = await handler.get()
        
        total_time = time.time() - start_time
        if enable_logs:
            print(f"Total generation time: {total_time:.2f}s")
        
        # Parse the result into our dataclass
        image_data = result['images'][0]
        return FluxProV1DepthResult(
            url=image_data['url'],
            width=image_data['width'],
            height=image_data['height'],
            content_type=image_data['content_type'],
            inference_time=result['timings'],
            seed=result['seed'],
            has_nsfw_concepts=result['has_nsfw_concepts'][0]
        )