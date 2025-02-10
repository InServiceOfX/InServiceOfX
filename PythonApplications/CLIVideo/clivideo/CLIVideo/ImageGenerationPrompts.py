from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from pathlib import Path
from morefal.FileIO import upload_image_to_fal
from morefal.Wrappers import FluxProV1Depth

class ImageGenerationPrompts:
    def __init__(self, session: PromptSession):
        self.session = session
        self._create_completer()

    def _create_completer(self):
        self.completer = WordCompleter([
            '.help',
            '.upload_image_to_fal',
            '.flux_v1_pro_depth',
            '.continue'
        ])

    async def handle_command(self, command: str) -> tuple[bool, str | None]:
        """Returns (should_continue, url_for_video)"""
        if command == '.help':
            print("\nAvailable commands:")
            print("  .upload_image_to_fal - Upload an image to fal.ai")
            print("  .flux_v1_pro_depth - Do image-to-image and style transfer using Flux Pro V1 Depth")
            print("  .continue - Continue with video generation")
            return True, None

        elif command == '.upload_image_to_fal':
            cwd = Path.cwd()
            print(f"\nCurrent working directory: {cwd}")
            print("Available files:")
            for file in cwd.glob('*'):
                if file.is_file():
                    print(f"  {file.name}")
            
            filename = await self.session.prompt_async("\nEnter image filename: ")
            try:
                image_url = upload_image_to_fal(cwd, filename)
                print(f"\nImage uploaded successfully!")
                print(f"URL: {image_url}")
                return True, image_url
            except FileNotFoundError:
                print(f"\nError: File '{filename}' not found")
                return True, None

        elif command == '.flux_v1_pro_depth':
            prompt = await self.session.prompt_async(
                "\nEnter prompt for image transformation: "
            )
            
            guidance_scale = await self.session.prompt_async(
                "\nEnter guidance scale (default: 3.5, lower values are more creative): "
            )
            try:
                guidance_scale = float(guidance_scale) if guidance_scale else 3.5
            except ValueError:
                print("Invalid guidance scale, using default 3.5")
                guidance_scale = 3.5

            image_url = await self.session.prompt_async(
                "\nEnter image URL (or press enter to upload new image): "
            )

            generator = FluxProV1Depth()
            result = await generator.generate(
                prompt=prompt,
                image_url=image_url,
                guidance_scale=guidance_scale
            )
            
            print(f"\nGeneration complete!")
            print(f"Result URL: {result.url}")
            print("full result: ", result)
            return True, result.url

        elif command == '.continue':
            return True, None

        return True, None