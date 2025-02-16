from morelumaai.Wrappers.ImageAndVideoManager import ImageAndVideoManager
from pathlib import Path
import yaml
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit import print_formatted_text
from prompt_toolkit.shortcuts.dialogs import yes_no_dialog

class UserExitActions:
    def __init__(self, config_path: Path):
        self.config_directory = config_path.parent
        self.available_images_file = self.config_directory / "available_images.yml"

    async def check_available_images_file(self) -> bool:
        """
        Check if images file exists and handle creation if needed
        
        Returns:
            bool: True if file exists or was created, False otherwise
        """
        if self.available_images_file.exists():
            print_formatted_text(HTML(
                f"\n<ansigreen>Found existing images file at: "
                f"{self.available_images_file}</ansigreen>"))
            return True

        should_create = await yes_no_dialog(
            title='Create Images File',
            text=f'No images file found at:\n{self.available_images_file}\n\nCreate new file?'
        ).run_async()

        if should_create:
            try:
                # Create empty YAML file
                with open(self.available_images_file, 'w') as f:
                    yaml.safe_dump([], f)
                print_formatted_text(HTML(
                    f"\n<ansigreen>Created new images file at: "
                    f"{self.available_images_file}</ansigreen>"))
                return True
            except Exception as e:
                print_formatted_text(HTML(
                    f"\n<ansired>Failed to create images file: {str(e)}</ansired>"))
                return False
        else:
            print_formatted_text(HTML(
                "\n<ansiyellow>Skipped creating images file</ansiyellow>"))
            return False

    def save_available_images(self, manager: ImageAndVideoManager) -> None:
        """Save unique available images to YAML file"""
        if not manager.available_images:
            return

        existing_images = []
        if self.available_images_file.exists():
            with open(self.available_images_file, 'r') as f:
                existing_images = yaml.safe_load(f) or []

        # Create new entries for unique URLs
        existing_urls = {img.get('url') for img in existing_images}
        new_images = []

        for image in manager.available_images:
            if image.url not in existing_urls:
                new_images.append({
                    'url': image.url,
                    'prompt_description': image.prompt_description
                })
                existing_urls.add(image.url)

        # Combine existing and new images
        all_images = existing_images + new_images

        # Save to file
        with open(self.available_images_file, 'w') as f:
            yaml.safe_dump(all_images, f)
