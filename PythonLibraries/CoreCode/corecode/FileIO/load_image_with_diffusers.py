from pathlib import Path
import diffusers

def load_image_with_diffusers(
    directory_with_file_or_full_path,
    file_name=None
    ):
    image_path=""
    if file_name == None:
        image_path=directory_with_file_or_full_path
    else:
        image_path = \
            Path(str(directory_with_file_or_full_path)) / str(file_name)
    return load_image(str(image_path))
