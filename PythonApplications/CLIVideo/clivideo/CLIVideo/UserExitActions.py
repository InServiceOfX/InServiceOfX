from morelumaai.Wrappers.ImageAndVideoManager import ImageAndVideoManager
from pathlib import Path
import yaml

class UserExitActions:
    def __init__(self, config_path: Path):
        self.config_directory = config_path.parent
        self.images_file = self.config_directory / "available_images.yml"

    def save_available_images(self, manager: ImageAndVideoManager) -> None:
        """Save unique available images to YAML file"""
        if not manager.available_images:
            return

        existing_images = []
        if self.images_file.exists():
            with open(self.images_file, 'r') as f:
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
        with open(self.images_file, 'w') as f:
            yaml.safe_dump(all_images, f)
