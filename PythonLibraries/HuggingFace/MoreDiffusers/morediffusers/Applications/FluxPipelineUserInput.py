from corecode.Utilities import (
    get_user_input,
    FloatParameter,
    IntParameter,
    StringParameter)
from morediffusers.Wrappers import create_seed_generator

from pathlib import Path


class FluxPipelineUserInput:

    def __init__(self, configuration):

        self.prompt = StringParameter(get_user_input(str, "Prompt: "))
        self.prompt_2 = StringParameter(get_user_input(str, "Prompt 2: ", ""))
        if self.prompt_2 == "":
            self.prompt_2 = None

        # Instead of calling this num_inference_steps to match the names in
        # diffusers' FluxPipeline, create_image_filename_and_save depends on
        # this.
        self.number_of_steps = IntParameter(
            get_user_input(int, "Number of inference steps, normally 28"))

        print("prompt: ", self.prompt.value)
        print("prompt_2: ", self.prompt_2.value)

        print("Number of (Inference) Steps: ", self.number_of_steps.value)

        self.base_filename = StringParameter(
            get_user_input(
                str,
                "Filename 'base', phrase common in the filenames"))

        self.iterations = IntParameter(
            get_user_input(int, "Number of Iterations: ", 2))

        self.model_name = Path(configuration.diffusion_model_path).name

        self.guidance_scale = FloatParameter(
            get_user_input(
                float,
                "Guidance Scale: ",
                7.0))

        self.guidance_scale = self.guidance_scale.value

        guidance_scale_step_explanation = \
            "Take the input guidance scale and input the value you would " + \
            "like to add or subtract (use - sign) to it upon each iteration;" + \
            " 7.0 + -0.5 = 6.5 on next iteration.\n"

        self.guidance_scale_step = FloatParameter(
            get_user_input(
                float,
                guidance_scale_step_explanation + \
                    "\nGuidance scale step value, enter small decimal value",
                0.0))

        self.num_images_per_prompt = IntParameter(
            get_user_input(
                int,
                "Number of images per prompt: ",
                1))

        # Set generator (for seed in torch), if it had been set in the configuration.
        self.generator = None
        if configuration.seed != None:

            generator = create_seed_generator(configuration, configuration.seed)

            self.generator = generator
