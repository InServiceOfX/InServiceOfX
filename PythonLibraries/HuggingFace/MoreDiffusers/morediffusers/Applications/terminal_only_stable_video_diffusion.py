from diffusers.utils import load_image
from pathlib import Path

import datetime
# For testing.
#import pickle
import sys
import time

python_libraries_path = Path(__file__).resolve().parents[4]
corecode_directory = python_libraries_path / "CoreCode"
more_computer_vision_directory = python_libraries_path / "MoreComputerVision"
more_diffusers_directory = \
    python_libraries_path / "HuggingFace" / "MoreDiffusers"

if not str(corecode_directory) in sys.path:
    sys.path.append(str(corecode_directory))

if not str(more_computer_vision_directory) in sys.path:
    sys.path.append(str(more_computer_vision_directory))

if not str(more_diffusers_directory) in sys.path:
    sys.path.append(str(more_diffusers_directory))

from corecode.Utilities import (
    clear_torch_cache_and_collect_garbage,
    get_user_input,
    StringParameter)

from morecomputervision.Conversions import export_to_mjpg_video

from morediffusers.Configurations import (
    GenerateVideoConfiguration,
    VideoConfiguration)

from morediffusers.Wrappers import (
    change_video_pipe_to_cuda_or_not,
    create_seed_generator,
    create_stable_video_diffusion_pipeline)


def terminal_only_stable_video_diffusion():

    start_time = time.time()

    configuration = VideoConfiguration()

    pipe = create_stable_video_diffusion_pipeline(
        configuration.diffusion_model_path,
        torch_dtype=configuration.torch_dtype,
        variant=configuration.variant,
        use_safetensors=configuration.use_safetensors,
        is_enable_cpu_offload=configuration.is_enable_cpu_offload,
        is_enable_sequential_cpu_offload=configuration.is_enable_sequential_cpu_offload)

    change_video_pipe_to_cuda_or_not(configuration, pipe)

    end_time = time.time()

    print("-------------------------------------------------------------------")
    print(f"Completed pipeline creation, took {end_time - start_time:.2f} seconds.")
    print("-------------------------------------------------------------------")

    try:
        print("\nDiagnostic: pipe.unet.config.num_frames: ")
        print(pipe.unet.config.num_frames)
    except AttributeError as err:
        print("AttributeError: ", err)

    generate_video_configuration = GenerateVideoConfiguration()

    image_prompt = load_image(generate_video_configuration.image_path)

    if generate_video_configuration.height != None and \
        generate_video_configuration.width != None:
        image_prompt.resize((
            generate_video_configuration.width,
            generate_video_configuration.height))

    generator = None

    if generate_video_configuration.seed != None:

        generator = create_seed_generator(
            configuration,
            generate_video_configuration.seed)

    start_time = time.time()

    # Of type 'list', of 'list''s, of length 1 typically or supposedly number of
    # videos (per prompt), typically default 1.
    # Each list is a list of PIL.Image.Image.
    frames_list = pipe(
        image=image_prompt,
        num_frames=generate_video_configuration.num_frames,
        # diffusers, pipeline_stable_video_diffusion.py
        # num_inference_steps (int, optional, defaults to 25)
        # number of denoising steps. More denoising steps usually lead to higher
        # quality video at expense of slower inference. This parameter is
        # modulated by 'strength'
        num_inference_steps=generate_video_configuration.num_inference_steps,
        min_guidance_scale=generate_video_configuration.min_guidance_scale,
        max_guidance_scale=generate_video_configuration.max_guidance_scale,
        fps=generate_video_configuration.fps,
        # motion_bucket_id ('int', optional, defaults to 127, int = 127)
        # Used for conditioning amount of motion for generation. Higher the
        # number the more motion will be in the video.
        motion_bucket_id=generate_video_configuration.motion_bucket_id,
        # noise_aug_strength (float, optional, defaults to 127)
        # noise_aug_strength: float = 0.02
        # Amount of noise added to init image, higher it is less the video will
        # look like init image. Increase it for more motion.
        noise_aug_strength=generate_video_configuration.noise_aug_strength,
        decode_chunk_size=generate_video_configuration.decode_chunk_size,
        num_videos_per_prompt=generate_video_configuration.num_videos_per_prompt,
        generator=generator,
        ).frames

    # Has length of 25 which seems to correspond to number of frames or number
    # of inference steps.
    frames = frames_list[0]

    end_time = time.time()

    print("-------------------------------------------------------------------")
    print(f"Completed frames generation, took {end_time - start_time:.2f} seconds.")
    print("-------------------------------------------------------------------")

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
        f"FPS{generate_video_configuration.fps}-{now}")

    file_path = Path(generate_video_configuration.temporary_save_path) / \
        f"{filename}.avi"

    # For testing.
    #frames_file_path = Path(generate_video_configuration.temporary_save_path) / \
    #    f"{filename}.pkl"
    #with open(frames_file_path, 'wb') as frames_file:
    #    pickle.dump(frames_list, frames_file)

    print("File path for file into export_to_video: ", str(file_path))

    try:
        output_video_path = export_to_mjpg_video(
            frames,
            str(file_path),
            fps=generate_video_configuration.fps)
        print("Exported!\n")
        print("Output video path from export_to_video: ", output_video_path)

    except RuntimeError as err:

        print(err.what())

    clear_torch_cache_and_collect_garbage()

if __name__ == "__main__":

    terminal_only_stable_video_diffusion()