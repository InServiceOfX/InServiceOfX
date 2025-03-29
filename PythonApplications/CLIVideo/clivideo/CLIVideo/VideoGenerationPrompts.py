from morelumaai.Wrappers.ImageAndVideoManager import ImageFrame
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit import print_formatted_text
from prompt_toolkit.shortcuts.dialogs import button_dialog, radiolist_dialog
from textwrap import shorten
from typing import Tuple, Optional

class VideoGenerationPrompts:
    # Define commands as class-level constants
    CMD_ADD_IMAGE = '.add_image'
    CMD_LIST_IMAGES = '.list_images'
    CMD_SET_START = '.set_start'
    CMD_SET_END = '.set_end'
    CMD_GENERATE = '.generate'
    CMD_LIST_GENERATIONS = '.list_generations'
    CMD_DELETE_GENERATION = '.delete_generation'
    CMD_HELP = '.help'
    CMD_EXIT = '.exit'
    CMD_LIST_GENERATION_IDS = '.list_generation_ids'
    
    # List of all commands in desired order
    COMMANDS = [
        CMD_ADD_IMAGE,
        CMD_LIST_IMAGES,
        CMD_SET_START,
        CMD_SET_END,
        CMD_GENERATE,
        CMD_LIST_GENERATIONS,
        CMD_DELETE_GENERATION,
        CMD_HELP,
        CMD_EXIT,
        CMD_LIST_GENERATION_IDS
    ]

    def __init__(self, prompt_modes, manager=None):
        self.prompt_modes = prompt_modes
        self.manager = manager
        self.commands = {
            self.CMD_ADD_IMAGE: self.handle_add_image,
            self.CMD_LIST_IMAGES: self.handle_list_images,
            self.CMD_SET_START: self.handle_set_start,
            self.CMD_SET_END: self.handle_set_end,
            self.CMD_GENERATE: self.handle_generate,
            self.CMD_LIST_GENERATIONS: self.handle_list_generations,
            self.CMD_DELETE_GENERATION: self.handle_delete_generation,
            self.CMD_HELP: self.show_help,
            self.CMD_EXIT: self.handle_exit,
            self.CMD_LIST_GENERATION_IDS: self.handle_list_generation_ids
        }

    async def handle_command(self, command: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Handle video generation commands
        
        Returns:
            Tuple[bool, Optional[str], Optional[str]]: 
            (should_continue, url, prompt_description)
        """
        command = command.strip().lower()
        if command in self.commands:
            return await self.commands[command]()
        return True, None, None

    async def handle_add_image(self) -> Tuple[bool, Optional[str], Optional[str]]:
        """Handle adding an image with optional description"""
        print("\n=== Add Image ===")
        url = await self.prompt_modes.prompt_async("Enter image URL: ")
        
        if not url.strip():
            print("No URL provided, canceling...")
            return True, None, None
            
        description = await self.prompt_modes.prompt_async(
            "Enter optional description (press Enter to skip): "
        )
        
        # Add the image to manager
        url = url.strip()
        desc = description.strip() if description.strip() else None
        self.manager.add_image(url, desc)
        print_formatted_text(HTML("\n<ansigreen>Image added successfully</ansigreen>"))
        
        return True, None, None

    async def handle_list_images(self) -> Tuple[bool, Optional[str], Optional[str]]:
        """Display all available images with their descriptions"""
        if not self.manager.available_images:
            print_formatted_text(
                HTML("\n<ansiyellow>No images available</ansiyellow>"))
            return True, None, None

        print_formatted_text(HTML("\n<ansigreen>Available Images:</ansigreen>"))
        for idx, image in enumerate(self.manager.available_images, 1):
            description = image.prompt_description or "No description"
            # Shorten description to fit on one line (adjust 60 as needed)
            short_desc = shorten(description, width=60, placeholder="...")
            
            # Format each entry with bullet point, index, URL and description
            entry = HTML(
                f"<ansired>•</ansired> <ansiblue>{idx}</ansiblue>. "
                f"URL: <ansiwhite>{image.url}</ansiwhite>\n"
                f"   Description: <ansiyellow>{short_desc}</ansiyellow>"
            )
            print_formatted_text(entry)
        print()  # Add blank line at end
        return True, None, None

    async def handle_set_start(
            self) -> Tuple[bool, Optional[str], Optional[str]]:
        """Handle setting the start frame"""
        # First show button dialog for frame type
        button_result = await button_dialog(
            title='Set Start Frame',
            text='Choose frame type:',
            buttons=[
                ('Image', 'image'),
                ('Cancel', None)
            ]
        ).run_async()
        
        if button_result is None:
            return True, None, None
        
        # Show current start frame status
        current = self.manager.start_frame
        status = "None"
        if current:
            if isinstance(current, ImageFrame):
                desc = current.prompt_description or "No description"
                status = (
                    f"Image: {shorten(current.url, 40)}... | "
                    f"{shorten(desc, 20)}")

        # Prepare radio list values
        values = [("none", "Clear start frame (None)")]

        # Add available images
        for idx, img in enumerate(self.manager.available_images):
            desc = img.prompt_description or "No description"
            label = f"Image {idx+1}: {shorten(desc, 40)}"
            values.append((str(idx), label))

        result = await radiolist_dialog(
            title="Set Start Frame",
            text=f"Current start frame: {status}\n\nSelect new start frame:",
            values=values
        ).run_async()

        if result is None:
            # Dialog cancelled
            return True, None, None

        if result == "none":
            self.manager.set_start_frame(None)
            print_formatted_text(
                HTML("\n<ansigreen>Start frame cleared</ansigreen>"))
        else:
            # Set selected image as start frame
            idx = int(result)
            self.manager.set_start_frame(
                self.manager.available_images[idx])
            print_formatted_text(HTML(
                f"\n<ansigreen>Start frame set to image {idx+1}</ansigreen>"))
        
        return True, None, None

    async def handle_set_end(self) -> Tuple[bool, Optional[str], Optional[str]]:
        """Handle setting the end frame"""
        # First show button dialog for frame type
        button_result = await button_dialog(
            title='Set End Frame',
            text='Choose frame type:',
            buttons=[
                ('Image', 'image'),
                ('Cancel', None)
            ]
        ).run_async()
        
        if button_result is None:
            return True, None, None
        
        # Show current end frame status
        current = self.manager.end_frame
        status = "None"
        if current:
            if isinstance(current, ImageFrame):
                desc = current.prompt_description or "No description"
                status = (
                    f"Image: {shorten(current.url, 40)}... | "
                    f"{shorten(desc, 20)}")

        # Prepare radio list values
        values = [("none", "Clear end frame (None)")]

        # Add available images
        for idx, img in enumerate(self.manager.available_images):
            desc = img.prompt_description or "No description"
            label = f"Image {idx+1}: {shorten(desc, 40)}"
            values.append((str(idx), label))

        result = await radiolist_dialog(
            title="Set End Frame",
            text=f"Current end frame: {status}\n\nSelect new end frame:",
            values=values
        ).run_async()

        if result is None:
            # Dialog cancelled
            return True, None, None

        if result == "none":
            self.manager.set_end_frame(None)
            print_formatted_text(
                HTML("\n<ansigreen>End frame cleared</ansigreen>"))
        else:
            # Set selected image as end frame
            idx = int(result)
            self.manager.set_end_frame(
                self.manager.available_images[idx])
            print_formatted_text(HTML(
                f"\n<ansigreen>End frame set to image {idx+1}</ansigreen>"))
        
        return True, None, None

    async def handle_generate(
            self) -> Tuple[bool, Optional[str], Optional[str]]:
        """Handle video generation with current keyframes"""
        print_formatted_text(
            HTML("\n<ansigreen>=== Video Generation ===</ansigreen>"))
        
        # Show current frame status
        start_status = "None"
        end_status = "None"
        
        if self.manager.start_frame:
            if isinstance(self.manager.start_frame, ImageFrame):
                desc = self.manager.start_frame.prompt_description or "No description"
                start_status = f"Image: {shorten(self.manager.start_frame.url, 40)}... | {shorten(desc, 20)}"
        
        if self.manager.end_frame:
            if isinstance(self.manager.end_frame, ImageFrame):
                desc = self.manager.end_frame.prompt_description or "No description"
                end_status = f"Image: {shorten(self.manager.end_frame.url, 40)}... | {shorten(desc, 20)}"
        
        print_formatted_text(HTML(
            f"\nCurrent frames:"
            f"\n<ansiyellow>Start frame:</ansiyellow> {start_status}"
            f"\n<ansiyellow>End frame:</ansiyellow> {end_status}\n"
        ))
        
        # Get generation prompt
        prompt = await self.prompt_modes.prompt_async(
            "Enter video generation prompt: "
        )
        
        if not prompt.strip():
            print_formatted_text(
                HTML("\n<ansired>No prompt provided, canceling...</ansired>"))
            return True, None, None
        
        print("\nGenerating video...")
        print_formatted_text(HTML("<ansiblue>Creating keyframes...</ansiblue>"))
        keyframes = self.manager.create_keyframes()
        
        if keyframes:
            print_formatted_text(HTML(
                f"<ansigreen>Keyframes created:</ansigreen> {keyframes}"))
            
            try:
                print_formatted_text(HTML("\n<ansiblue>Calling Luma AI API...</ansiblue>"))
                result = self.manager.generate(prompt)
                
                if result:
                    print_formatted_text(HTML(
                        f"\n<ansigreen>Success! Video URL:</ansigreen> {result}"))
                    # Save the video
                    try:
                        save_path = self.manager.save_video()
                        print_formatted_text(HTML(
                            f"\n<ansigreen>Video saved to:</ansigreen> {save_path}"))
                    except Exception as e:
                        print_formatted_text(HTML(
                            f"\n<ansired>Failed to save video: {str(e)}</ansired>"))
                else:
                    print_formatted_text(HTML(
                        "\n<ansired>Generation failed or returned no result</ansired>"))
            except Exception as e:
                print_formatted_text(HTML(
                    f"\n<ansired>Error during generation: {str(e)}</ansired>"))
        else:
            print_formatted_text(HTML(
                "\n<ansired>No keyframes available. "
                "Please set at least one frame (start or end) first.</ansired>"))
        
        return True, None, None

    async def handle_list_generations(self) -> Tuple[bool, Optional[str], Optional[str]]:
        """Handle listing all available generations"""
        print_formatted_text(HTML("\n<ansigreen>=== Available Generations ===</ansigreen>"))
        
        # Update generations list
        try:
            self.manager.update_generations_list()
        except Exception as e:
            print_formatted_text(HTML(
                f"\n<ansired>Failed to fetch generations: {str(e)}</ansired>"))
            return True, None, None
        
        if not self.manager.parsed_generations:
            print_formatted_text(HTML("\n<ansiyellow>No generations found</ansiyellow>"))
            return True, None, None
        
        # Format each generation into a list of strings
        formatted_items = []
        for gen in self.manager.parsed_generations.values():
            created_str = gen.created_at.strftime("%Y-%m-%d %H:%M") if gen.created_at else "N/A"
            # Always show first 2 chars + [..] to indicate truncation
            id_short = gen.id[:2] + "[..]"
            prompt_short = shorten(gen.request_prompt or "No prompt", 40)
            
            item = (
                f"<ansiyellow>{id_short}</ansiyellow> | "
                f"{created_str} | "
                f"{gen.assets_video or 'No video'} | "
                f"{prompt_short}"
            )
            formatted_items.append(item)
        
        # Print header
        print_formatted_text(HTML(
            "\n<ansiblue>ID     | Created At       | Video URL | Prompt</ansiblue>"
            "\n" + "-" * 100
        ))
        
        # Print items
        for item in formatted_items:
            print_formatted_text(HTML(item))
        
        print()  # Add blank line at end
        return True, None, None

    async def handle_list_generation_ids(self) -> Tuple[bool, Optional[str], Optional[str]]:
        """Handle listing all available generations with full IDs"""
        print_formatted_text(HTML("\n<ansigreen>=== Available Generations (Full IDs) ===</ansigreen>"))
        
        # Update generations list
        try:
            self.manager.update_generations_list()
        except Exception as e:
            print_formatted_text(HTML(
                f"\n<ansired>Failed to fetch generations: {str(e)}</ansired>"))
            return True, None, None
        
        if not self.manager.parsed_generations:
            print_formatted_text(HTML("\n<ansiyellow>No generations found</ansiyellow>"))
            return True, None, None
        
        # Format each generation into a list of strings
        formatted_items = []
        for gen in self.manager.parsed_generations.values():
            created_str = gen.created_at.strftime("%Y-%m-%d %H:%M") if gen.created_at else "N/A"
            # Show full ID (no truncation)
            id_full = gen.id
            # Significantly shorten prompt to make room for full ID
            prompt_short = shorten(gen.request_prompt or "No prompt", 20)
            
            item = (
                f"<ansiyellow>{id_full}</ansiyellow> | "
                f"{created_str} | "
                f"{gen.assets_video or 'No video'} | "
                f"{prompt_short}"
            )
            formatted_items.append(item)
        
        # Print header with adjusted column widths for full IDs
        print_formatted_text(HTML(
            "\n<ansiblue>ID                                       | Created At       | Video URL | Prompt</ansiblue>"
            "\n" + "-" * 120
        ))
        
        # Print items
        for item in formatted_items:
            print_formatted_text(HTML(item))
        
        print()  # Add blank line at end
        return True, None, None

    async def handle_delete_generation(self) -> Tuple[bool, Optional[str], Optional[str]]:
        """Handle deletion of a generation"""
        print_formatted_text(HTML("\n<ansigreen>=== Delete Generation ===</ansigreen>"))
        
        # Update generations list first
        try:
            self.manager.update_generations_list()
        except Exception as e:
            print_formatted_text(HTML(
                f"\n<ansired>Failed to fetch generations: {str(e)}</ansired>"))
            return True, None, None
        
        if not self.manager.parsed_generations:
            print_formatted_text(HTML("\n<ansiyellow>No generations found to delete</ansiyellow>"))
            return True, None, None
        
        # Prepare radio list values
        values = []
        for gen in self.manager.parsed_generations.values():
            id_short = gen.id[:8] + "..." + gen.id[-4:]  # Show first 8 and last 4 chars
            created_str = gen.created_at.strftime("%Y-%m-%d %H:%M") if gen.created_at else "N/A"
            prompt_short = shorten(gen.request_prompt or "No prompt", 30)
            
            # Extract filename from video URL or show "No video"
            video_info = "No video"
            if gen.assets_video:
                try:
                    from urllib.parse import urlparse
                    video_path = urlparse(gen.assets_video).path
                    filename = video_path.split('/')[-1]
                    video_info = shorten(filename, 20)
                except:
                    video_info = shorten(gen.assets_video, 20)
            
            label = (
                f"ID: {id_short} | "
                f"Created: {created_str} | "
                f"Video: {video_info} | "
                f"Prompt: {prompt_short}"
            )
            values.append((gen.id, label))
        
        result = await radiolist_dialog(
            title="Delete Generation",
            text="Select a generation to delete (or press Esc to cancel):",
            values=values
        ).run_async()
        
        if result is None:
            print_formatted_text(HTML("\n<ansiyellow>Deletion cancelled</ansiyellow>"))
            return True, None, None
        
        try:
            self.manager.generate_video.delete_generation(result)
            print_formatted_text(HTML(
                f"\n<ansigreen>Successfully deleted generation: {result}</ansigreen>"))
            # Update the list after deletion
            self.manager.update_generations_list()
        except Exception as e:
            print_formatted_text(HTML(
                f"\n<ansired>Failed to delete generation: {str(e)}</ansired>"))
        
        return True, None, None

    async def show_help(self) -> Tuple[bool, Optional[str], Optional[str]]:
        """Show available commands"""
        print("\nAvailable commands:")
        print("  .add_image         - Add an image with optional description")
        print("  .list_images       - Show all available images")
        print("  .list_generations  - Show all available generations")
        print("  .list_generation_ids  - Show all available generation IDs")
        print("  .delete_generation - Delete a specific generation")
        print("  .set_start         - Set or clear the start frame")
        print("  .set_end           - Set or clear the end frame")
        print("  .generate          - Generate video with current frames")
        print("  .help             - Show this help message")
        print("  .exit             - Exit the program")
        return True, None, None

    async def handle_exit(self) -> Tuple[bool, Optional[str], Optional[str]]:
        """Handle exit command with save functionality"""
        if self.manager.available_images:
            file_ready = await self.prompt_modes.user_exit_actions.check_available_images_file()
            if file_ready:
                self.prompt_modes.user_exit_actions.save_available_images(
                    self.manager)
        
        print_formatted_text(HTML("\n<ansigreen>Goodbye!</ansigreen>"))
        return False, None, None
