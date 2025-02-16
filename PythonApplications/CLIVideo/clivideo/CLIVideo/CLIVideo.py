from pathlib import Path
from prompt_toolkit import PromptSession
from prompt_toolkit.styles import Style
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.shortcuts import clear
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit import print_formatted_text
from prompt_toolkit.shortcuts.dialogs import yes_no_dialog

from clivideo.Utilities import get_environment_variable

from morelumaai.Configuration import GenerationConfiguration
from morelumaai.Wrappers.ImageAndVideoManager import ImageAndVideoManager
from morelumaai.Wrappers.get_camera_motions import get_camera_motions
from clivideo.CLIVideo.ImageGenerationPrompts import ImageGenerationPrompts
import asyncio

from clivideo.Configuration.CLIVideoConfiguration import CLIVideoConfiguration
from clivideo.CLIVideo.PromptModes import PromptModes, PromptMode
from clivideo.CLIVideo.VideoGenerationPrompts import VideoGenerationPrompts
from clivideo.CLIVideo.UserStartActions import UserStartActions
from clivideo.CLIVideo.UserExitActions import UserExitActions

class CLIVideo:
    def __init__(
        self,
        clivideo_configuration_path: Path,
        lumaai_configuration_path: Path
    ):
        self.configuration = CLIVideoConfiguration(clivideo_configuration_path)
        self.manager = ImageAndVideoManager(
            GenerationConfiguration.from_yaml(lumaai_configuration_path))
        self.user_start_actions = UserStartActions(lumaai_configuration_path.parent)
        self.user_exit_actions = UserExitActions(lumaai_configuration_path)
        self.prompt_modes = PromptModes(self.configuration)
        self.video_generation_prompts = VideoGenerationPrompts(
            self.prompt_modes,
            self.manager)
        self.image_generation_prompts = ImageGenerationPrompts(
            self.prompt_modes)
        
        # Add user_exit_actions to prompt_modes for access in commands
        self.prompt_modes.user_exit_actions = self.user_exit_actions

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
            print_formatted_text(HTML(
                f"\n<ansired>Warning: Could not fetch camera motions: {str(e)}</ansired>\n"
            ))

    async def run_iterative(self):
        """Single iteration of video generation"""
        try:
            prompt = await self.prompt_modes.session.prompt_async(
                "Video prompt (or type .help for options): "
            )
            
            if not prompt.strip():
                return True
                
            if prompt.startswith('.'):
                if self.prompt_modes.current_mode == PromptMode.IMAGE_VIDEO:
                    should_continue, url, desc = await self.video_generation_prompts.handle_command(prompt)
                else:  # IMAGE_GENERATION mode
                    should_continue, url, desc = await self.image_generation_prompts.handle_command(prompt)
                return should_continue

            self.last_prompt = prompt
            self.prompt_history.append(prompt)
            
            return True
            
        except KeyboardInterrupt:
            # Handle Ctrl+C exit
            if self.manager.available_images:
                file_ready = await self.user_exit_actions.check_available_images_file()
                if file_ready:
                    self.user_exit_actions.save_available_images(self.manager)
            print_formatted_text(HTML("\n<ansigreen>Goodbye!</ansigreen>"))
            return False
        except Exception as e:
            print(f"\nError: {str(e)}\n")
            return True

    def run(self):
        """Main run loop"""
        clear()
        print("Welcome to CLIVideo! Press Ctrl+C to exit.\n")
        self.user_start_actions.load_available_images(self.manager)

        async def run_async():
            continue_running = True
            while continue_running:
                continue_running = await self.run_iterative()
        
        asyncio.run(run_async())
        print("\nThank you for using CLIVideo!")