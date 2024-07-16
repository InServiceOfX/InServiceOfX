from corecode.Utilities import (
    get_user_input,
    FloatParameter,
    IntParameter,
    StringParameter)

from pathlib import Path

import torch

class UserInputWithLoras:

    def __init__(self, configuration, loras_configuration):

        self.prompt = StringParameter(get_user_input(str, "Prompt: "))
        self.prompt_2 = StringParameter(get_user_input(str, "Prompt 2: ", ""))
        if self.prompt_2 == "":
            self.prompt_2 = None

        # Example negative prompt:
        # "(lowres, low quality, worst quality:1.2), (text:1.2), glitch, deformed, mutated, cross-eyed, ugly, disfigured (lowres, low quality, worst quality:1.2), (text:1.2), watermark, painting, drawing, illustration, glitch,deformed, mutated, cross-eyed, ugly, disfigured"
        # prompt for what you want to not include.
        self.negative_prompt = StringParameter(
            get_user_input(str, "Negative prompt: ", ""))

        self.negative_prompt_2 = StringParameter(
            get_user_input(str, "Negative prompt 2: ", ""))
        if self.negative_prompt_2 == "":
            self.negative_prompt_2 = None

        self.number_of_steps = IntParameter(
            get_user_input(int, "Number of steps, normally 50"))

        print("prompt: ", self.prompt.value)
        print("negative prompt: ", self.negative_prompt.value)
        print("Number of Steps: ", self.number_of_steps.value)

        self.base_filename = StringParameter(
            get_user_input(
                str,
                "Filename 'base', phrase common in the filenames"))

        self.iterations = IntParameter(
            get_user_input(int, "Number of Iterations: ", 2))

        self.model_name = Path(configuration.diffusion_model_path).name

        self.guidance_scale = configuration.guidance_scale
        self.guidance_scale_step = 0.0

        # Assume that if guidance scale was indeed set in the configuration,
        # then the user has intention of changing it.
        if self.guidance_scale is not None:

            self.guidance_scale_step = FloatParameter(
                get_user_input(
                    float,
                    "Guidance scale step value, enter small decimal value",
                    0.0))

        self.cross_attention_kwargs=None
        
        if loras_configuration.lora_scale != None:
            self.cross_attention_kwargs={
                "scale": float(loras_configuration.lora_scale)}

        # Set generator (for seed in torch), if it had been set in the configuration.
        self.generator = None
        if configuration.seed != None:

            seed = int(configuration.seed)

            if configuration.is_enable_cpu_offload == False and \
                configuration.is_enable_sequential_cpu_offload == False:
                # TODO: determine where the generator should be.
                #and \
                #(configuration.is_to_cuda == False or \
                #    configuration.is_to_cuda == None):
                generator = torch.Generator(device='cpu')
                generator.manual_seed(seed)
            else:
                # https://pytorch.org/docs/stable/generated/torch.Generator.html
                generator = torch.Generator(device='cuda')
                generator.manual_seed(seed)
            self.generator = generator
