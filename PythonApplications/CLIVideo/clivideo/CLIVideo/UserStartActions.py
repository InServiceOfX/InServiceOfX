from pathlib import Path
import yaml

class UserStartActions:
    def __init__(self, config_path: Path):
        self.config_directory = config_path.parent
        self.images_file = self.config_directory / "available_images.yml"

    def load_available_images(self, manager) -> None:
        """Load available images from YAML file if it exists"""
        if not self.images_file.exists():
            return

        try:
            with open(self.images_file, 'r') as f:
                saved_images = yaml.safe_load(f) or []

            # Track existing URLs to avoid duplicates
            existing_urls = {img.url for img in manager.available_images}

            for image in saved_images:
                if image['url'] not in existing_urls:
                    manager.add_image(
                        url=image['url'],
                        prompt_description=image.get('prompt_description')
                    )
                    existing_urls.add(image['url'])
        except Exception as e:
            print(f"Error loading available images: {str(e)}")
