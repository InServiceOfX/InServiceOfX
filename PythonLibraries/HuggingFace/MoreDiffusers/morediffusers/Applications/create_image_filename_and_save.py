from corecode.Utilities.Strings import format_float_for_string

from pathlib import Path

def create_image_filename_and_save(user_input, index, image, configuration):
    """
    @param user_input UserInputWithLoras instance
    @param index int
    An arbitrary integer, typically in a sequence, to enumerate the image
    filename.
    """

    filename = ""

    if user_input.guidance_scale is None:

        filename = (
            f"{user_input.base_filename.value}{user_input.model_name}-"
            f"Steps{user_input.number_of_steps.value}Iter{index}"
        )
    else:

        filename = (
            f"{user_input.base_filename.value}{user_input.model_name}-"
            f"Steps{user_input.number_of_steps.value}Iter{index}Guidance{format_float_for_string(user_input.guidance_scale)}"
        )

    image_format = image.format if image.format else "PNG"

    file_path = Path(configuration.temporary_save_path) / \
        f"{filename}.{image_format.lower()}"
    image.save(file_path)
    print(f"Image saved to {file_path}")

    return filename, file_path