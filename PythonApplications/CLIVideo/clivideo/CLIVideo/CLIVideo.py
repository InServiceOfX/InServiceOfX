from pathlib import Path
from prompt_toolkit import PromptSession
from prompt_toolkit.styles import Style
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.shortcuts import clear

from clivideo.Utilities import get_environment_variable

from morelumaai.Configuration import GenerationConfiguration
from morelumaai.Wrappers import (GenerateVideo, get_camera_motions)
from clivideo.CLIVideo.ImageGenerationPrompts import ImageGenerationPrompts
import asyncio

class CLIVideo:
    def __init__(
        self,
        configuration_path: Path,
        lumaai_configuration_path: Path):
        self.generation_configuration = GenerationConfiguration.from_yaml(
            lumaai_configuration_path)
        self.generator = GenerateVideo(
            self.generation_configuration,
            get_environment_variable("LUMAAI_API_KEY"))
        self.prompt_history = []
        self.last_prompt = None
        
        # Get and display camera motions
        try:
            camera_motions = get_camera_motions(
                get_environment_variable("LUMAAI_API_KEY"))
            print(
                ("\033[1m\n🎥 VIDEO GENERATION TIP: Use these EXACT camera "
                 "motion keywords in your prompt! 🎥\033[0m"))
            # Print in columns of 3
            for i in range(0, len(camera_motions), 3):
                row = camera_motions[i:i+3]
                print("  ".join(f"'{motion}'" for motion in row))
            print("\n\033[1mExample: 'Push In on a grand estate...'\033[0m\n")
        except Exception as e:
            print(f"\nWarning: Could not fetch camera motions: {str(e)}\n")
        
        # CLI Styling
        self.prompt_style = Style.from_dict({
            'prompt': '#00aa00 bold',
            'continuation': 'gray'
        })
        
        self.session = PromptSession(
            history=InMemoryHistory(),
            style=self.prompt_style,
            wrap_lines=True
        )

        self.image_generation_prompts = ImageGenerationPrompts(self.session)

    async def run_iterative(self):
        """Single iteration of video generation"""
        try:
            # Get main prompt
            prompt = await self.session.prompt_async(
                "Video prompt (or type .help for options): ",
                completer=self.image_generation_prompts.completer
            )
            
            if not prompt.strip():
                return True
                
            if prompt.startswith('.'):
                should_continue, url = await self.image_generation_prompts.handle_command(
                    prompt)
                if url:
                    # Use the generated image as start frame
                    keyframes = self.generator.create_start_keyframe(url)
                return should_continue

            self.last_prompt = prompt
            self.prompt_history.append(prompt)

            # Get optional start frame - make async
            start_url = await self.session.prompt_async(
                "Start frame URL (optional, press Enter to skip): "
            )
            start_url = start_url.strip()
            
            keyframes = None
            if start_url:
                keyframes = self.generator.create_start_keyframe(start_url)
                # Only ask for end frame if start frame was provided
                end_url = await self.session.prompt_async(
                    "End frame URL (optional, press Enter to skip): "
                )
                end_url = end_url.strip()
                if end_url:
                    keyframes.update(self.generator.create_end_keyframe(end_url))

            print("keyframes: ", keyframes)

            # Generate and save video
            print("\nGenerating video...")
            if keyframes is not None:
                self.generator.generate(prompt, keyframes)
            else:
                self.generator.generate(prompt)
            save_path = self.generator.save_video()
            print(f"\nVideo saved to: {save_path}\n")

            return True

        except KeyboardInterrupt:
            return False
        except Exception as e:
            print(f"\nError: {str(e)}\n")
            return True

    def run(self):
        """Main run loop"""
        clear()
        print("Welcome to CLIVideo! Press Ctrl+C to exit.\n")
        
        async def run_async():
            continue_running = True
            while continue_running:
                continue_running = await self.run_iterative()
        
        asyncio.run(run_async())
        print("\nThank you for using CLIVideo!")