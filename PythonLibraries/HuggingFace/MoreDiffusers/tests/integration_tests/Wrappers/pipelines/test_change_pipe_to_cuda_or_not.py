from morediffusers.Configurations import (
    DiffusionPipelineConfiguration,
    StableDiffusionXLGenerationConfiguration,
    )
from morediffusers.Wrappers.pipelines import (
    create_stable_diffusion_xl_pipeline,
    change_pipe_to_cuda_or_not)

from pathlib import Path

test_data_directory = Path(__file__).resolve().parents[3] / "TestData"

def test_change_pipe_to_cuda_or_not_works_with_configuration():
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
            "Create detailed and hyper realistic portrait of a woman. "
            "focal lenght 80 milimeters, aperture 1.2. "
            "Shallow depth of field. Medium format feel. "
            "Very detailed and crisp image. "
            "Lady has delicate face. "
            "Large blue wide open eyes. "
            "Slightly open lips with bit seductive expression. "
            "Light is coming from a side, through the window. "
            "She sits on a bench leaning against the white rendered bit uneven wall."
        ),
        "negative_prompt": "not ugly."
    }

    generation_kwargs.update(more_generation_kwargs)

    image = pipe(**generation_kwargs).images[0]

    save_path = Path.cwd() / "test_change_pipe_to_cuda_or_not_output.png"
    image.save(str(save_path))