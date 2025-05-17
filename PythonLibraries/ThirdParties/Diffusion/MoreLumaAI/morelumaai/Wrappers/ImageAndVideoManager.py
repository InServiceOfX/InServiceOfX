from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from morelumaai.Configuration import GenerationConfiguration
from morelumaai.Wrappers import GenerateVideo
from pathlib import Path
from typing import Optional, List, Dict, Union

class FrameType(Enum):
    IMAGE = "image"
    GENERATION = "generation"

class FramePosition(Enum):
    START = "frame0"
    END = "frame1"

@dataclass
class ImageFrame:
    url: str
    prompt_description: Optional[str] = None
    type: str = FrameType.IMAGE.value

@dataclass
class GenerationFrame:
    id: str
    prompt_description: Optional[str] = None
    type: str = FrameType.GENERATION.value

@dataclass
class ParsedGeneration:
    id: str
    assets_image: Optional[str]
    assets_video: Optional[str]
    created_at: Optional[datetime]
    request_callback_url: Optional[str]
    request_prompt: Optional[str]

class ImageAndVideoManager:
    def __init__(
            self,
            configuration: GenerationConfiguration,
            api_key: str = None):
        self.generate_video = GenerateVideo(configuration, api_key)
        self.available_images: List[ImageFrame] = []
        # Using OrderedDict to maintain insertion order and ensure unique IDs
        self.available_generations: OrderedDict[str, GenerationFrame] = \
            OrderedDict()
        self.start_frame: Optional[Union[ImageFrame, GenerationFrame]] = None
        self.end_frame: Optional[Union[ImageFrame, GenerationFrame]] = None
        self._current_generations_list = None
        self.parsed_generations: OrderedDict[str, ParsedGeneration] = OrderedDict()

    def generate(self, prompt: str):
        """Generate a video using the available frames"""
        keyframes = self.create_keyframes()
        if keyframes is None or len(keyframes) == 0:
            print("No keyframes to generate video")
            return None
        return self.generate_video.generate(prompt, keyframes)  
    
    def save_video(self) -> Path:
        """Save the generated video to disk"""
        return self.generate_video.save_video()

    def set_loop(self, loop: bool = False):
        """Set the loop property of the generate video object"""
        self.generate_video.set_loop(loop)

    def add_image(
            self,
            url: str,
            prompt_description: Optional[str] = None) -> None:
        """Add an image to the available images list"""
        self.available_images.append(
            ImageFrame(url=url, prompt_description=prompt_description))

    def set_start_frame(
            self,
            frame: Optional[Union[ImageFrame, GenerationFrame]] = None) -> None:
        """Set or clear the start frame (frame0)"""
        self.start_frame = frame

    def set_end_frame(
            self,
            frame: Optional[Union[ImageFrame, GenerationFrame]] = None) -> None:
        """Set or clear the end frame (frame1)"""
        self.end_frame = frame

    def clear_frames(self) -> None:
        """Clear both start and end frames"""
        self.start_frame = None
        self.end_frame = None

    def _add_generation(
            self,
            generation_id: str,
            prompt_description: Optional[str] = None) -> None:
        """Add a generation to available generations if ID doesn't exist"""
        if generation_id not in self.available_generations:
            self.available_generations[generation_id] = GenerationFrame(
                id=generation_id,
                prompt_description=prompt_description
            )

    def _get_generation_by_id(
            self,
            generation_id: str) -> Optional[GenerationFrame]:
        """Get a generation by its ID"""
        return self.available_generations.get(generation_id)

    def _get_generations_in_order(self) -> List[GenerationFrame]:
        """Get all generations in the order they were added"""
        return list(self.available_generations.values())

    def create_keyframes(self) -> Dict:
        """Create the keyframes dictionary based on current start and end frames"""
        keyframes = {}
        
        if self.start_frame:
            if isinstance(self.start_frame, ImageFrame):
                keyframes[FramePosition.START.value] = {
                    "type": FrameType.IMAGE.value,
                    "url": self.start_frame.url
                }
            else:  # GenerationFrame
                keyframes[FramePosition.START.value] = {
                    "type": FrameType.GENERATION.value,
                    "id": self.start_frame.id
                }

        if self.end_frame:
            if isinstance(self.end_frame, ImageFrame):
                keyframes[FramePosition.END.value] = {
                    "type": FrameType.IMAGE.value,
                    "url": self.end_frame.url
                }
            else:  # GenerationFrame
                keyframes[FramePosition.END.value] = {
                    "type": FrameType.GENERATION.value,
                    "id": self.end_frame.id
                }

        return keyframes

    def update_generations_list(self, limit: int = 100) -> None:
        """Update the list of generations and parse them into OrderedDict"""
        self._current_generations_list = \
            self.generate_video.list_all_generations(limit)
        
        # Clear and rebuild parsed generations
        self.parsed_generations.clear()
        self.available_generations.clear()

        for gen in self._current_generations_list.generations:
            parsed = ParsedGeneration(
                id=gen.id,
                assets_image=gen.assets.image if gen.assets else None,
                assets_video=gen.assets.video if gen.assets else None,
                created_at=gen.created_at,
                request_callback_url=gen.request.callback_url \
                    if gen.request.callback_url else None,
                request_prompt=gen.request.prompt \
                    if gen.request.prompt else None
            )
            self.parsed_generations[gen.id] = parsed

        # Verify count matches
        assert len(self.parsed_generations) == \
            self._current_generations_list.count, \
            "Parsed generation count doesn't match API response count"

        for gen in self.parsed_generations.values():
            self._add_generation(gen.id, gen.request_prompt)

        assert len(self.available_generations) == len(self.parsed_generations), \
            "Available generations count doesn't match parsed generations count"


    def _create_formatted_generations_list(self) -> List[Dict]:
        """
        Get a formatted list of parsed generations with human-readable dates
        and simplified structure
        
        Returns:
            List[Dict]: List of dictionaries containing formatted generation
            info
        """
        formatted_list = []
        
        for generation in self.parsed_generations.values():
            formatted_entry = {
                "id": generation.id,
                "video_url": generation.assets_video,
                "created_at": generation.created_at.strftime(
                    "%Y-%m-%d %H:%M:%S") \
                    if generation.created_at else None,
                "prompt": generation.request_prompt
            }
            formatted_list.append(formatted_entry)
        
        return formatted_list

        
