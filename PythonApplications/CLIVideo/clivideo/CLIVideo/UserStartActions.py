from pathlib import Path
import yaml
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit import print_formatted_text

class UserStartActions:
    def __init__(self, config_path: Path):
        self.config_directory = config_path
        self.available_images_file = self.config_directory / "available_images.yml"

    def load_available_images(self, manager) -> None:
        """Load available images from YAML file if it exists"""
        if not self.available_images_file.exists():
            print_formatted_text(HTML(
                "\n<ansiyellow>No saved images file found at: "
                f"{self.available_images_file}</ansiyellow>"))
            return

        try:
            with open(self.available_images_file, 'r') as f:
                saved_images = yaml.safe_load(f) or []

            if not saved_images:
                print_formatted_text(HTML(
                    "\n<ansiyellow>No saved images found in file</ansiyellow>"))
                return

            # Track existing URLs to avoid duplicates
            existing_urls = {img.url for img in manager.available_images}
            loaded_count = 0

            for image in saved_images:
                if image['url'] not in existing_urls:
                    manager.add_image(
                        url=image['url'],
                        prompt_description=image.get('prompt_description')
                    )
                    existing_urls.add(image['url'])
                    loaded_count += 1

            if loaded_count > 0:
                print_formatted_text(HTML(
                    f"\n<ansigreen>Successfully loaded {loaded_count} "
                    f"image{'s' if loaded_count > 1 else ''}</ansigreen>"))
            else:
                print_formatted_text(HTML(
                    "\n<ansiyellow>No new images to load</ansiyellow>"))

        except Exception as e:
            print_formatted_text(HTML(
                f"\n<ansired>Error loading available images: {str(e)}</ansired>"))
