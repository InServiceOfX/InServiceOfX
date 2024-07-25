from diffusers.utils import load_image
from pathlib import Path

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

from morediffusers.Applications import create_video_file_path

from morediffusers.Configurations import (
    GenerateVideoConfiguration,
    VideoConfiguration)

from morediffusers.Wrappers import (
    change_video_pipe_to_cuda_or_not,
    create_seed_generator,
    create_i2vgen_xl_pipeline)


def terminal_only_iv2gen_xl():

    start_time = time.time()

    configuration = VideoConfiguration()

    pipe = create_i2vgen_xl_pipeline(
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

    print("\nDiagnostic: pipe.unet.config.num_frames: ")

    try:
        print(pipe.unet.config.num_frames)
    except AttributeError as err:
        print(
            "AttributeError for calling pipe.unet.config.num_frames",
            err)

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

    if generate_video_configuration.height == None:
        generate_video_configuration.height = 704
    if generate_video_configuration.width == None:
        generate_video_configuration.width = 1280
    if generate_video_configuration.num_frames == None:
        generate_video_configuration.num_frames = 16
    if generate_video_configuration.guidance_scale == None:
        generate_video_configuration.guidance_scale = 9.0
    if generate_video_configuration.clip_skip == None:
        generate_video_configuration.clip_skip = 1

    prompt = StringParameter(get_user_input(str, "Prompt: "))
    negative_prompt = StringParameter(get_user_input(
        str,
        "Negative prompt: ",
        ""))

    print("prompt: ", prompt.value)
    print("negative prompt: ", negative_prompt.value)

    start_time = time.time()

    frames_list = pipe(
        prompt=prompt.value,
        image=image_prompt,
        # Optional[int] = 16
        # Rate at which generated images shall be exported to a video after
        # generation. This is also used as a "micro-condition" while generation.
        target_fps=generate_video_configuration.fps,
        # int = 16
        # The number of video frames to generate.
        num_frames=generate_video_configuration.num_frames,
        # The number of denoising steps.
        num_inference_steps=generate_video_configuration.num_inference_steps,
        guidance_scale=generate_video_configuration.guidance_scale,
        negative_prompt=negative_prompt.value,
        num_videos_per_prompt=generate_video_configuration.num_videos_per_prompt,
        # Optional[int] = 1
        # Number of frames to decode at a time. Higher the chunk size, higher
        # the temporary consistency between frames, but also higher the memory
        # consumption.
        decode_chunk_size=generate_video_configuration.decode_chunk_size,
        generator=generator,
        # Number of layers to be skipped from CLIP while computing prompt
        # embeddings. Value of 1 means output of pre-final layer will be used.
        clip_skip=generate_video_configuration.clip_skip
        ).frames

    frames = frames_list[0]

    end_time = time.time()

    print("-------------------------------------------------------------------")
    print(f"Completed frames generation, took {end_time - start_time:.2f} seconds.")
    print("-------------------------------------------------------------------")

    file_path = create_video_file_path(
        configuration,
        generate_video_configuration)

    #frames_list_file_path = file_path.with_suffix(
    #    file_path.suffix + '.pkl' if file_path.suffix else '.pkl')

    #with open(frames_list_file_path, 'wb') as frames_file:
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

        print("RuntimeError: ", err)

    except AttributeError as err:
        print("AttributeError: ", err)

    clear_torch_cache_and_collect_garbage()

if __name__ == "__main__":

    terminal_only_iv2gen_xl()