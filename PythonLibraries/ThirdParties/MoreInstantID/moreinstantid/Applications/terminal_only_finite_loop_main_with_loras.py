"""
@brief Generate N (hence "finite") number of images in a for loop (hence
"loop"). Run this in your terminal, command prompt (hence "terminal only").
"""

from collections import namedtuple
from pathlib import Path
import sys
import time

python_libraries_path = Path(__file__).resolve().parents[4]
corecode_directory = python_libraries_path / "CoreCode"
morecomputervision_directory = python_libraries_path / "MoreComputerVision"
more_diffusers_directory = \
    python_libraries_path / "HuggingFace" / "MoreDiffusers"
more_insightface_directory = \
    python_libraries_path / "ThirdParties" / "MoreInsightFace"
more_instant_id_directory = \
    python_libraries_path / "ThirdParties" / "MoreInstantID"

instant_id_directory = python_libraries_path.parent.parent / "ThirdParty" / \
    "InstantID"

if not str(corecode_directory) in sys.path:
    sys.path.append(str(corecode_directory))
if not str(morecomputervision_directory) in sys.path:
    sys.path.append(str(morecomputervision_directory))
if not str(more_diffusers_directory) in sys.path:
    sys.path.append(str(more_diffusers_directory))
if not str(more_insightface_directory) in sys.path:
    sys.path.append(str(more_insightface_directory))
if not str(more_instant_id_directory) in sys.path:
    sys.path.append(str(more_instant_id_directory))
if not str(instant_id_directory) in sys.path:
    sys.path.append(str(instant_id_directory))

from corecode.Utilities import (
    clear_torch_cache_and_collect_garbage,
    get_user_input,
    FloatParameter,
    IntParameter,
    StringParameter)
from corecode.Utilities.Strings import format_float_for_string

from morediffusers.Applications import (
    print_loras_diagnostics,
    print_pipeline_diagnostics,
    UserInputWithLoras
)

from morediffusers.Configurations import Configuration as PipelineConfiguration
from morediffusers.Configurations import LoRAsConfigurationForMoreDiffusers as \
    LoRAsConfiguration

from morediffusers.Wrappers import load_loras

from morediffusers.Schedulers import change_scheduler_or_not
from moreinsightface.Wrappers import get_face_and_pose_info_from_images
from moreinstantid.Wrappers import (
    create_controlnet,
    create_stable_diffusion_xl_pipeline,
    generate_image_with_loras,
)
from moreinstantid.Configuration import Configuration


def terminal_only_finite_loop_main_with_loras():

    start_time = time.time()

    pipeline_configuration = PipelineConfiguration()
    configuration = Configuration()

    face_information, pose_information = get_face_and_pose_info_from_images(
        model_name=configuration.face_analysis_model_name,
        model_root_directory=str(
            configuration.face_analysis_model_directory_path),
        face_image_path=configuration.face_image_path,
        pose_image_path=configuration.pose_image_path,
        det_size=configuration.det_size)
    configuration.det_size

    end_time = time.time()
    duration = end_time - start_time

    print("-------------------------------------------------------------------")
    print("Completed initialization, obtained configuration, face, post information.")
    print(f"Took {duration:.2f} seconds to initialize.")
    print("-------------------------------------------------------------------")

    start_time = time.time()

    # The ControlNet binary offered by InstantID doesn't seem to work with
    # 16-bit float type from torch because on CPU, there's no 16-bit float type.
    # So we didn't add the torch_dtype argument.
    controlnet = create_controlnet(configuration.control_net_model_path)
    pipe = create_stable_diffusion_xl_pipeline(
        controlnet,
        pipeline_configuration.diffusion_model_path,
        configuration.ip_adapter_path,
        is_enable_cpu_offload=True,
        is_enable_sequential_cpu=True)

    original_scheduler_name = pipe.scheduler.config._class_name

    is_scheduler_changed = change_scheduler_or_not(
        pipe,
        pipeline_configuration.scheduler)

    end_time = time.time()

    print_pipeline_diagnostics(
        end_time - start_time,
        pipe,
        is_scheduler_changed,
        original_scheduler_name)

    print(
        "\nDiagnostic: pipe.unet.config.time_cond_proj_dim used in pipeline_stable_diffusion_xl_instantid to determine optionally getting Guidance Scale Embedding or not: ")

    print(pipe.unet.config.time_cond_proj_dim)

    # This was set by running create_stable_diffusion_xl_pipeline, which then
    # ran _encode_prompt_image_emb(..) for StableDiffusionXLInstantID class.
    print(
        "\nDiagnostic: pipe.image_proj_model_in_features used in pipeline_stable_diffusion_xl_instantid in _encode_prompt_image_emb: ")

    print(pipe.image_proj_model_in_features)

    #
    #
    # LoRAs - Low Rank Adaptations
    #
    #
    start_time = time.time()

    loras_configuration = LoRAsConfiguration()
    load_loras(pipe, loras_configuration)

    end_time = time.time()

    print_loras_diagnostics(end_time - start_time, pipe)

    #
    #
    # END of LoRAs - Low Rank Adaptations
    #
    #

    user_input = UserInputWithLoras(pipeline_configuration, loras_configuration)


    ip_adapter_scale = FloatParameter(
        get_user_input(
            float,
            "IP Adapter Scale: Enter value from 0 to 1.5, normally 0.8, but enter 'base' value"
            ))

    controlnet_conditioning_scale = FloatParameter(
        get_user_input(
            float,
            "ControlNet Conditioning: normally 0.8 or 1.0, enter 'base' value"))

    print("IP adapter scale: ", ip_adapter_scale.value)
    print("ControlNet Conditioning: ", controlnet_conditioning_scale.value)

    ip_adapter_step = FloatParameter(
        get_user_input(
            float,
            "IP Adapter step value, enter small decimal value",
            0.0))

    controlnet_conditioning_step = FloatParameter(
        get_user_input(
            float,
            "ControlNet conditioning step value, enter small decimal value",
            0.0))

    ip_adapter_scale_value = ip_adapter_scale.value
    controlnet_conditioning_scale_value = controlnet_conditioning_scale.value

    for index in range(user_input.iterations.value):

        image = generate_image_with_loras(
            pipe,
            prompt=user_input.prompt.value,
            face_information=face_information,
            prompt_2=user_input.prompt_2.value,
            negative_prompt=user_input.negative_prompt.value,
            negative_prompt_2=user_input.negative_prompt_2.value,
            pose_information=pose_information,
            ip_adapter_scale=ip_adapter_scale_value,
            controlnet_conditioning_scale=controlnet_conditioning_scale_value,
            number_of_steps=user_input.number_of_steps.value,
            guidance_scale=user_input.guidance_scale,
            lora_scale=loras_configuration.lora_scale,
            clip_skip=pipeline_configuration.clip_skip,
            generator=user_input.generator
            )

        filename = ""

        if user_input.guidance_scale is None:

            filename = (
                f"{user_input.base_filename.value}{user_input.model_name}-IPAdapter{format_float_for_string(ip_adapter_scale_value)}"
                f"ControlNet{format_float_for_string(controlnet_conditioning_scale_value)}"
                f"Steps{user_input.number_of_steps.value}Iter{index}"
            )
        else:

            filename = (
                f"{user_input.base_filename.value}{user_input.model_name}-IPAdapter{format_float_for_string(ip_adapter_scale_value)}"
                f"ControlNet{format_float_for_string(controlnet_conditioning_scale_value)}"
                f"Steps{user_input.number_of_steps.value}Iter{index}Guidance{format_float_for_string(user_input.guidance_scale)}"
            )

        image_format = image.format if image.format else "PNG"
        file_path = Path(pipeline_configuration.temporary_save_path) / \
            f"{filename}.{image_format.lower()}"
        image.save(file_path)
        print(f"Image saved to {file_path}")

        # Update parameters for iterative steps.
        ip_adapter_scale_value += ip_adapter_step.value
        controlnet_conditioning_scale_value += \
            controlnet_conditioning_step.value

        if user_input.guidance_scale is not None:
            user_input.guidance_scale += user_input.guidance_scale_step.value

    clear_torch_cache_and_collect_garbage()


if __name__ == "__main__":

    terminal_only_finite_loop_main_with_loras()