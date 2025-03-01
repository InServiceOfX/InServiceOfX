from morediffusers.Configurations import (
    DiffusionPipelineConfiguration,
    LoRAsConfigurationForMoreDiffusers,
    StableDiffusionXLGenerationConfiguration)
from morediffusers.Wrappers.pipelines import (
    create_stable_diffusion_xl_pipeline,
    change_pipe_to_cuda_or_not,
    load_loras)

from pathlib import Path

import torch

test_data_directory = Path(__file__).resolve().parents[2] / "TestData"

def test_generator_with_seed_works_on_cuda_device():
    test_file_path = test_data_directory / "sdxl_pipeline_configuration.yml"

    configuration = DiffusionPipelineConfiguration(test_file_path)

    pipe = create_stable_diffusion_xl_pipeline(configuration)

    change_pipe_to_cuda_or_not(configuration, pipe)

    test_file_path = test_data_directory / "sdxl_generation_configuration.yml"

    generation_configuration = StableDiffusionXLGenerationConfiguration(test_file_path)

    generation_kwargs = generation_configuration.get_generation_kwargs()

    # https://civitai.com/images/20900686
    more_generation_kwargs = {
        "prompt": (
            "cinematic film still of young teen ((ohwx woman)),tanned skin,"
            "centered in the image, peeking out from under a blanket fort in the"
            "afternoon, flash polaroid photo by george hurrell, hazy light rays,"
            "golden hour,35mm photograph,(looking at viewer),super detailed, UHD,"
            "petite body, bokeh, professional, shallow depth of field,8k uhd,"
            "dslr, soft lighting, high quality, film grain, shallow depth of field,"
            "vignette, highly detailed, high budget, bokeh, cinemascope, moody,"
            "epic, gorgeous, film grain, grainy"),
        "negative_prompt": "(3d render:1.1), cleavage, nude"
    }

    generation_kwargs.update(more_generation_kwargs)

    # Create generator with correct device
    generator = torch.Generator(device=configuration.cuda_device)
    generator.manual_seed(generation_configuration.seed)

    generation_kwargs["generator"] = generator

    image = pipe(**generation_kwargs).images[0]

    save_path = Path.cwd() / "test_seed_generator_output_1.png"
    image.save(str(save_path))

def test_generator_with_seed_works_on_cuda_device_with_loras():
    test_file_path = test_data_directory / "sdxl_pipeline_configuration.yml"

    configuration = DiffusionPipelineConfiguration(test_file_path)

    pipe = create_stable_diffusion_xl_pipeline(configuration)

    change_pipe_to_cuda_or_not(configuration, pipe)

    test_file_path = test_data_directory / "sdxl_loras_configuration.yml"

    loras_configuration = LoRAsConfigurationForMoreDiffusers(test_file_path)

    load_loras(pipe, loras_configuration)

    test_file_path = test_data_directory / "sdxl_generation_configuration.yml"

    generation_configuration = StableDiffusionXLGenerationConfiguration(test_file_path)

    generation_kwargs = generation_configuration.get_generation_kwargs()

    # https://civitai.com/images/5344331
    more_generation_kwargs = {
        "prompt": (
            "cute woman Jennifer with brown hair at burning man fire, "
            "daylight, harsh shadow, dusty wind,  steampunk cosplay, "
            "(freckles:0.2)"
        ),
        "negative_prompt": "(3d render:1.1), cleavage, nude"
    }

    generation_kwargs.update(more_generation_kwargs)

    # Create generator with correct device
    generator = torch.Generator(device=configuration.cuda_device)
    generator.manual_seed(generation_configuration.seed)

    generation_kwargs["generator"] = generator

    image = pipe(**generation_kwargs).images[0]

    save_path = Path.cwd() / "test_seed_generator_output_2.png"
    image.save(str(save_path))