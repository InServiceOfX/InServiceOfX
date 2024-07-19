from pathlib import Path
from typing import (List, Union)

# Empirically, it was found that you needed to do both these imports.
import PIL
from PIL import Image

import numpy as np
import tempfile

def export_to_mjpg_video(
    video_frames: Union[List[np.ndarray], List[PIL.Image.Image]],
    output_video_path: str = None,
    fps: int = 10) -> str:
    """
    @brief Export frames into a MJPEG-encoded video.
    @details See
    https://stackoverflow.com/questions/64506236/pil-image-list-into-a-video-slide-with-cv2-videowriter
    The cv2.resize(image, ..) step was helpful.
    """
    import cv2

    if output_video_path is None:
        output_video_path = tempfile.NamedTemporaryFile(suffix=".avi").name
    else:
        # Check if suffix is '.avi'
        path = Path(output_video_path)
        if path.suffix != '.avi':
            path = path.with_suffix('.avi')
            output_video_path = str(path)

    if isinstance(video_frames[0], np.ndarray):
        video_frames = [
            (frame * 255).astype(np.uint8) for frame in video_frames]

    elif isinstance(video_frames[0], PIL.Image.Image):
        video_frames = [np.array(frame) for frame in video_frames]

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    w, h, c = video_frames[0].shape
    video_writer = cv2.VideoWriter(
        output_video_path,
        fourcc,
        fps=fps,
        frameSize=(w, h))
    for i in range(len(video_frames)):
        img = cv2.cvtColor(video_frames[i], cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, (w, h))
        video_writer.write(img)

    video_writer.release()

    if not Path(output_video_path).exists():
        raise RuntimeError("Output file was not created: ", output_video_path)

    return output_video_path
