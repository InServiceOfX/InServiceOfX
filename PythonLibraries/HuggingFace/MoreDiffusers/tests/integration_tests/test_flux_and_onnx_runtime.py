from corecode.Utilities import (
    DataSubdirectories,
    )

from optimum.onnxruntime import ORTFluxPipeline

from morediffusers.Configurations import DiffusionPipelineConfiguration

from pathlib import Path

import torch

data_sub_dirs = DataSubdirectories()
test_data_directory = Path(__file__).resolve().parents[1] / "TestData"

pretrained_onnx_model_name_or_path = \
    data_sub_dirs.ModelsDiffusion / "black-forest-labs" / "FLUX.1-dev-onnx"

if not pretrained_onnx_model_name_or_path.exists():

    pretrained_onnx_model_name_or_path = \
        Path("/Data1/Models/Diffusion/") / "black-forest-labs" / "FLUX.1-dev-onnx"

pretrained_diffusion_model_name_or_path = \
    data_sub_dirs.ModelsDiffusion / "black-forest-labs" / "FLUX.1-dev"

if not pretrained_diffusion_model_name_or_path.exists():

    pretrained_diffusion_model_name_or_path = \
        Path("/Data1/Models/Diffusion/") / "black-forest-labs" / "FLUX.1-dev"

def test_ORTFluxPipeline_from_pretrained():

    test_file_path = test_data_directory / "flux_pipeline_configuration_empty.yml"
    configuration = DiffusionPipelineConfiguration(test_file_path)

    configuration.diffusion_model_path = pretrained_onnx_model_name_or_path
    configuration.torch_dtype = torch.bfloat16
    configuration.cuda_device = "cuda:0"
    configuration.is_to_cuda = True

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

    # E           ValueError: The library name could not be automatically inferred. If using the command-line, please provide the argument --library {transformers,diffusers,timm,sentence_transformers}. Example: `--library diffusers`.
    #
    # /usr/local/lib/python3.10/dist-packages/optimum/exporters/tasks.py:2026: ValueError
    pipeline = ORTFluxPipeline.from_pretrained(
        str(pretrained_diffusion_model_name_or_path),
        **pipeline_kwargs)


def test_optimum_pipeline():
    from optimum.pipelines import pipeline

    # E       AttributeError: 'NoneType' object has no attribute 'startswith'
    # /usr/local/lib/python3.10/dist-packages/optimum/pipelines/pipelines_base.py:297: AttributeError    

    pipeline = pipeline(model=pretrained_onnx_model_name_or_path,)

def test_ORTFluxPipeline_as_ORTDiffusionPipeline_in_steps():
    import onnxruntime as ort

    transformer_fp4_path = pretrained_onnx_model_name_or_path / \
        "transformer.opt" / "fp4"

    assert transformer_fp4_path.exists()

    # onnxruntime.capi.onnxruntime_pybind11_state.InvalidProtobuf: [ONNXRuntimeError] : 7 : INVALID_PROTOBUF : Load model from /Data1/Models/Diffusion/black-forest-labs/FLUX.1-dev-onnx/transformer.opt/fp4 failed:Protobuf parsing failed.
    # /usr/local/lib/python3.10/dist-packages/onnxruntime/capi/onnxruntime_inference_collection.py:526: InvalidProtobuf
    transformer_fp4_session = ort.InferenceSession(str(transformer_fp4_path))

def test_ORTFluxPipeline_from_pretrained_with_export():

    test_file_path = test_data_directory / "flux_pipeline_configuration_empty.yml"
    configuration = DiffusionPipelineConfiguration(test_file_path)

    configuration.diffusion_model_path = pretrained_onnx_model_name_or_path
    configuration.cuda_device = "cuda:0"
    configuration.is_to_cuda = True

    pipeline_kwargs = configuration.get_pretrained_kwargs()

    pipeline = ORTFluxPipeline.from_pretrained(
        str(pretrained_diffusion_model_name_or_path),
        **pipeline_kwargs,
        export=True,
        )