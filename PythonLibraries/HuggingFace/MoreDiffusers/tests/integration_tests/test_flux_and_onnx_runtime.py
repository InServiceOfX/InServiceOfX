from corecode.Utilities import (
    DataSubdirectories,
    )

from optimum.onnxruntime import ORTFluxPipeline

from morediffusers.Configurations import DiffusionPipelineConfiguration

from pathlib import Path

import torch

data_sub_dirs = DataSubdirectories()
test_data_directory = Path(__file__).resolve().parents[1] / "TestData"

def test_ORTFluxPipeline_from_pretrained():

    test_file_path = test_data_directory / "flux_pipeline_configuration_empty.yml"
    configuration = DiffusionPipelineConfiguration(test_file_path)

    pretrained_diffusion_model_name_or_path = \
        data_sub_dirs.ModelsDiffusion / "black-forest-labs" / "FLUX.1-dev-onnx"

    if not pretrained_diffusion_model_name_or_path.exists():

        pretrained_diffusion_model_name_or_path = \
            Path("/Data1/Models/Diffusion/") / "black-forest-labs" / "FLUX.1-dev-onnx"

    configuration.diffusion_model_path = pretrained_diffusion_model_name_or_path
    configuration.torch_dtype = torch.bfloat16
    configuration.cuda_device = "cuda:0"

    # from diffusers import FluxPipeline
    # and
    # class ORTFluxPipeline(ORTDiffusionPipeline, FluxPipeline)
    # and defined with no __init__(), but inherits from those classes, including
    # ORTDiffusionPipeline.
    # In optimum/onnxruntime/modeling_diffusion.py
    # class ORTDiffusionPipeline(ORTModel, DiffusionPipeline)
    # def __init__(self, .. , **kwargs,)
    # and kwargs is used in self.shared_attributes_init(..)
    # class ORTModel(OptimizedModel):
    # def shared_attributes_init(self, ..)

    pipeline_kwargs = configuration.get_pretrained_kwargs()
    pipeline_kwargs["variant"] = "fp8"

    pipeline = ORTFluxPipeline.from_pretrained(
        str(pretrained_diffusion_model_name_or_path),
        **pipeline_kwargs)
