from pathlib import Path
import datetime

from corecode.Utilities import (
    get_user_input,
    StringParameter)


def create_video_file_path(
    configuration,
    generate_video_configuration,
    suffix=".avi") -> Path:
    """
    @param user_input UserInputWithLoras instance
    @param index int
    An arbitrary integer, typically in a sequence, to enumerate the image
    filename.
    """

    base_filename = StringParameter(
        get_user_input(
            str,
            "Filename 'base', phrase common in the filenames"))

    model_name = Path(configuration.diffusion_model_path).name

    filename = ""

    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    filename = (
        f"{base_filename.value}{model_name}-"
        f"Steps{generate_video_configuration.num_inference_steps}-"
        f"FPS{generate_video_configuration.fps}-{now}{suffix}")

    file_path = Path(generate_video_configuration.temporary_save_path) / \
        f"{filename}"

    return file_path
