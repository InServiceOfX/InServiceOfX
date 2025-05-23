from corecode.Utilities.Strings import format_float_for_string

from pathlib import Path

def create_image_filenames_and_save_images(
    user_input,
    index,
    images,
    configuration):
    """
    @param user_input UserInputWithLoras instance
    @param index int
    An arbitrary integer, typically in a sequence, to enumerate the image
    filename.
    """

    filename_start = ""

    if user_input.guidance_scale is None:

        filename_start = (
            f"{user_input.base_filename.value}{user_input.model_name}-"
            f"Steps{user_input.number_of_steps.value}Iter{index}"
        )
    else:

        filename_start = (
            f"{user_input.base_filename.value}{user_input.model_name}-"
            f"Steps{user_input.number_of_steps.value}Iter{index}Guidance{format_float_for_string(user_input.guidance_scale)}"
        )

    number_of_images = len(images)

    for i in range(number_of_images):
        filename = filename_start + f"Image{i}"

        image_format = images[i].format if images[i].format else "PNG"

        file_path = Path(configuration.temporary_save_path) / \
            f"{filename}.{image_format.lower()}"
        images[i].save(file_path)
        print(f"Image saved to {file_path}")

    return filename_start, file_path