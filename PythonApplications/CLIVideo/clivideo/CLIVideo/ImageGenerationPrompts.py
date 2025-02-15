from prompt_toolkit.completion import WordCompleter
from pathlib import Path
from morefal.FileIO import upload_image_to_fal
from morefal.Wrappers import FluxProV1Depth
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit import print_formatted_text
from typing import Tuple, Optional
from clivideo.CLIVideo.PromptModes import (
    PromptModes,
    PromptMode)
import re

class ImageGenerationPrompts:
    def __init__(self, prompt_modes: PromptModes):
        self.prompt_modes = prompt_modes
        self.commands = {
            '.upload': self.handle_upload_image_to_fal,
            '.transform': self.handle_flux_v1_pro_depth,
            '.help': self.show_help
        }
        
        # Add commands to PromptModes completers
        dot_pattern = re.compile(r'^\.')
        self.prompt_modes.completers[PromptMode.IMAGE_GENERATION] = WordCompleter(
            list(self.commands.keys()), 
            pattern=dot_pattern)
        
    async def handle_command(
            self,
            command: str) -> Tuple[bool, Optional[str], Optional[str]]:
        command = command.strip().lower()
        if command in self.commands:
            return await self.commands[command]()

    async def handle_upload_image_to_fal(
            self) -> Tuple[bool, Optional[str], Optional[str]]:
        cwd = Path.cwd()
        print_formatted_text(
            HTML(f"\n<ansiblue>Current working directory: {cwd}</ansiblue>"))
        print_formatted_text(
            HTML("<ansigreen>Available files:</ansigreen>"))
        for file in cwd.glob('*'):
            if file.is_file():
                print(f"  {file.name}")
        
        filename = await self.prompt_modes.prompt_async("\nEnter image filename: ")
        try:
            image_url = upload_image_to_fal(cwd, filename)
            print_formatted_text(HTML("\n<ansigreen>Image uploaded successfully!</ansigreen>"))
            print_formatted_text(HTML(f"URL: <ansiblue>{image_url}</ansiblue>"))
            return True, image_url, None
        except FileNotFoundError:
            print_formatted_text(HTML(f"\n<ansired>Error: File '{filename}' not found</ansired>"))
            return True, None, None

    async def handle_flux_v1_pro_depth(self) -> Tuple[bool, Optional[str], Optional[str]]:
        prompt = await self.prompt_modes.prompt_async(
            "\nEnter prompt for image transformation: ")
        
        guidance_scale = await self.prompt_modes.prompt_async(
            "\nEnter guidance scale (default: 3.5, lower values are more creative): "
        )
        try:
            guidance_scale = float(guidance_scale) if guidance_scale else 3.5
        except ValueError:
            print_formatted_text(HTML(
                "<ansiyellow>Invalid guidance scale, using default 3.5</ansiyellow>"
            ))
            guidance_scale = 3.5
        image_url = await self.prompt_modes.prompt_async(
            "\nEnter image URL (or press enter to upload new image): "
        )

        generator = FluxProV1Depth()
        result = await generator.generate(
            prompt=prompt,
            image_url=image_url,
            guidance_scale=guidance_scale
        )
        
        print_formatted_text(
            HTML("\n<ansigreen>Generation complete!</ansigreen>"))
        print_formatted_text(
            HTML(f"Result URL: <ansiblue>{result.url}</ansiblue>"))
        return True, result.url, None

    async def show_help(self) -> Tuple[bool, Optional[str], Optional[str]]:
        print_formatted_text(HTML("\n<ansigreen>Available commands:</ansigreen>"))
        print_formatted_text(HTML("  <ansiblue>.upload</ansiblue>    - Upload a local image to Fal AI"))
        print_formatted_text(HTML("  <ansiblue>.transform</ansiblue>  - Transform an image using Flux Pro V1"))
        print_formatted_text(HTML("  <ansiblue>.help</ansiblue>      - Show this help message"))
        return True, None, None