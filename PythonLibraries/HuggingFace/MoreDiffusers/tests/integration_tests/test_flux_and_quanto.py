from corecode.Utilities import (
    DataSubdirectories,
    )

from morediffusers.Configurations import DiffusionPipelineConfiguration
from morediffusers.Wrappers.pipelines import (
    change_pipe_to_cuda_or_not,
    create_flux_pipeline,
    )

from diffusers import FluxPipeline
from transformers import T5EncoderModel

from pathlib import Path
import json
import torch

data_sub_dirs = DataSubdirectories()
test_data_directory = Path(__file__).resolve().parents[1] / "TestData"

pretrained_diffusion_model_name_or_path = \
    data_sub_dirs.ModelsDiffusion / "black-forest-labs" / "FLUX.1-dev"
if not pretrained_diffusion_model_name_or_path.exists():

    pretrained_diffusion_model_name_or_path = \
        Path("/Data1/Models/Diffusion/") / "black-forest-labs" / "FLUX.1-dev"

from optimum.quanto import freeze, qint8, qint4, quantize, quantization_map

from diffusers.models import FluxTransformer2DModel
from optimum.quanto import QuantizedDiffusersModel, QuantizedTransformersModel

class QuantizedFluxTransformer2DModel(QuantizedDiffusersModel):
    base_class = FluxTransformer2DModel

class QuantizedT5EncoderModelForCausalLM(QuantizedTransformersModel):
    auto_class = T5EncoderModel
    auto_class.from_config = auto_class._from_config

def test_flux_dev_quanto_qint8():

    test_file_path = test_data_directory / "flux_pipeline_configuration_empty.yml"
    configuration = DiffusionPipelineConfiguration(test_file_path)
    configuration.diffusion_model_path = pretrained_diffusion_model_name_or_path
    configuration.torch_dtype = torch.bfloat16
    configuration.cuda_device = "cuda:0"
    configuration.is_to_cuda = True

    pipeline = create_flux_pipeline(configuration)
    quantize(pipeline.transformer, weights=qint8, exclude="proj_out")
    quantize(pipeline.text_encoder_2, weights=qint8)
    freeze(pipeline.transformer)
    freeze(pipeline.text_encoder_2)

def test_flux_save_qint8_transformer():

    test_file_path = test_data_directory / "flux_pipeline_configuration_empty.yml"
    configuration = DiffusionPipelineConfiguration(test_file_path)
    configuration.diffusion_model_path = pretrained_diffusion_model_name_or_path
    configuration.torch_dtype = torch.bfloat16
    configuration.cuda_device = "cuda:0"
    configuration.is_to_cuda = True
    model_kwargs = configuration.get_pretrained_kwargs()

    model = FluxTransformer2DModel.from_pretrained(
        pretrained_diffusion_model_name_or_path,
        subfolder="transformer",
        **model_kwargs,
        )
    qmodel = QuantizedDiffusersModel.quantize(
        model,
        weights=qint8,
        exclude="proj_out",
        )

    qmodel.save_pretrained("flux-dev-transformer-qint8")

def test_flux_save_qint8_text_encoder_2():
    test_file_path = test_data_directory / "flux_pipeline_configuration_empty.yml"
    configuration = DiffusionPipelineConfiguration(test_file_path)
    configuration.diffusion_model_path = pretrained_diffusion_model_name_or_path
    configuration.torch_dtype = torch.bfloat16
    configuration.cuda_device = "cuda:0"
    configuration.is_to_cuda = True
    model_kwargs = configuration.get_pretrained_kwargs()

    text_encoder_2 = T5EncoderModel.from_pretrained(
        pretrained_diffusion_model_name_or_path,
        subfolder="text_encoder_2",
        **model_kwargs,
        )
    quantize(text_encoder_2, weights=qint8)
    freeze(text_encoder_2)
    text_encoder_2.save_pretrained("flux-dev-text-encoder-2-qint8")
    qmap = quantization_map(text_encoder_2)
    f = open("quanto_qmap.json", "w", encoding="utf8")
    json.dump(qmap, f)
    f.close()

def test_flux_load_transformer_qint8():
    test_file_path = test_data_directory / "flux_pipeline_configuration_empty.yml"
    configuration = DiffusionPipelineConfiguration(test_file_path)
    configuration.torch_dtype = torch.bfloat16

    path = pretrained_diffusion_model_name_or_path.parents[0] / \
        "flux-dev-transformer-qint8"

    quantized_transformer = QuantizedFluxTransformer2DModel.from_pretrained(
        str(path)).to(configuration.torch_dtype)

def test_flux_load_text_encoder_2_qint8():
    test_file_path = test_data_directory / "flux_pipeline_configuration_empty.yml"
    configuration = DiffusionPipelineConfiguration(test_file_path)
    configuration.torch_dtype = torch.bfloat16

    path = pretrained_diffusion_model_name_or_path.parents[0] / \
        "flux-dev-text-encoder-2-qint8"

    quantized_text_encoder_2 = QuantizedTransformersModel.from_pretrained(
        str(path)).to(configuration.torch_dtype)

def test_flux_create_pipeline_qint8():
    test_file_path = test_data_directory / "flux_pipeline_configuration_empty.yml"
    configuration = DiffusionPipelineConfiguration(test_file_path)
    configuration.diffusion_model_path = pretrained_diffusion_model_name_or_path
    configuration.torch_dtype = torch.bfloat16
    configuration.cuda_device = "cuda:0"
    configuration.is_to_cuda = True

    path = pretrained_diffusion_model_name_or_path.parents[0] / \
        "flux-dev-transformer-qint8"

    quantized_transformer = QuantizedFluxTransformer2DModel.from_pretrained(
        str(path)).to(configuration.torch_dtype)

    path = pretrained_diffusion_model_name_or_path.parents[0] / \
        "flux-dev-text-encoder-2-qint8"

    quantized_text_encoder_2 = \
        QuantizedT5EncoderModelForCausalLM.from_pretrained(
            str(path)).to(configuration.torch_dtype)

    pipeline = create_flux_pipeline(configuration)
    pipeline.transformer = quantized_transformer
    pipeline.text_encoder_2 = quantized_text_encoder_2

    pipeline = change_pipe_to_cuda_or_not(configuration, pipeline)

def test_flux_save_qint4_transformer():

    test_file_path = test_data_directory / "flux_pipeline_configuration_empty.yml"
    configuration = DiffusionPipelineConfiguration(test_file_path)
    configuration.diffusion_model_path = pretrained_diffusion_model_name_or_path
    configuration.torch_dtype = torch.bfloat16
    configuration.cuda_device = "cuda:0"
    configuration.is_to_cuda = True
    model_kwargs = configuration.get_pretrained_kwargs()
    model_kwargs = configuration.get_pretrained_kwargs()
    assert len(model_kwargs) == 1
    assert model_kwargs["torch_dtype"] == torch.bfloat16 

    transformer = FluxTransformer2DModel.from_pretrained(
        pretrained_diffusion_model_name_or_path,
        subfolder="transformer",
        **model_kwargs,
        )
    # qmodel = QuantizedDiffusersModel.quantize(
    #     model,
    #     weights=qint4,
    #     exclude="proj_out",
    #     )

    #qmodel.save_pretrained("flux-dev-transformer-qint4")
    #quantize(transformer, weights=qint4, exclude="proj_out")
    quantize(transformer, weights=qint4)
    freeze(transformer)

    transformer.save_pretrained("flux-dev-transformer-bfloat16-qint4")

    qmap = quantization_map(transformer)
    with open("quanto_qmap.json", "w", encoding="utf8") as f:
        json.dump(qmap, f, indent=4)


def test_flux_save_qint4_text_encoder_2():
    test_file_path = test_data_directory / "flux_pipeline_configuration_empty.yml"
    configuration = DiffusionPipelineConfiguration(test_file_path)
    configuration.diffusion_model_path = pretrained_diffusion_model_name_or_path
    configuration.torch_dtype = torch.bfloat16
    configuration.cuda_device = "cuda:0"
    configuration.is_to_cuda = True
    model_kwargs = configuration.get_pretrained_kwargs()

    text_encoder_2 = T5EncoderModel.from_pretrained(
        pretrained_diffusion_model_name_or_path,
        subfolder="text_encoder_2",
        **model_kwargs,
                                                                                                                                                                                                                                                                                )
    quantize(text_encoder_2, weights=qint4)
    freeze(text_encoder_2)
    text_encoder_2.save_pretrained("flux-dev-text-encoder-2-bfloat16-qint4")
    qmap = quantization_map(text_encoder_2)
    with open("quanto_qmap.json", "w", encoding="utf8") as f:
        json.dump(qmap, f, indent=4)

def test_flux_create_pipeline_qint4():
    test_file_path = test_data_directory / "flux_pipeline_configuration_empty.yml"
    configuration = DiffusionPipelineConfiguration(test_file_path)
    configuration.diffusion_model_path = pretrained_diffusion_model_name_or_path
    configuration.torch_dtype = torch.bfloat16
    # TODO: Change from bfloat16?
    configuration.cuda_device = "cuda:0"
    configuration.is_to_cuda = True

    path = pretrained_diffusion_model_name_or_path.parents[0] / \
        "flux-dev-transformer-bfloat16-qint4"

    assert path.exists()

    # quantized_transformer = QuantizedFluxTransformer2DModel.from_pretrained(
    #     str(path)).to(
    #         #device=configuration.cuda_device,
    #         dtype=configuration.torch_dtype)

    transformer = FluxTransformer2DModel.from_pretrained(
        str(path)).to(dtype=configuration.torch_dtype)

    path = pretrained_diffusion_model_name_or_path.parents[0] / \
        "flux-dev-text-encoder-2-bfloat16-qint4"

    assert path.exists()

    quantized_text_encoder_2 = \
        QuantizedT5EncoderModelForCausalLM.from_pretrained(
            str(path)).to(
                #device=configuration.cuda_device,
                dtype=configuration.torch_dtype)

    # pipeline = create_flux_pipeline(configuration)
    # pipeline.transformer = quantized_transformer
    # pipeline.text_encoder_2 = quantized_text_encoder_2

    # pipeline = change_pipe_to_cuda_or_not(configuration, pipeline)

    pipeline = FluxPipeline.from_pretrained(
        pretrained_diffusion_model_name_or_path,
        transformer=transformer,
        text_encoder_2=quantized_text_encoder_2,
        torch_dtype=torch.bfloat16)
    pipeline = pipeline.to(configuration.cuda_device)

# Try
# https://github.com/huggingface/optimum-quanto/issues/270

def test_flux_save_float16_qint4_transformer():

    test_file_path = test_data_directory / "flux_pipeline_configuration_empty.yml"
    configuration = DiffusionPipelineConfiguration(test_file_path)
    configuration.diffusion_model_path = pretrained_diffusion_model_name_or_path
    configuration.torch_dtype = torch.float16
    configuration.cuda_device = "cuda:0"
    configuration.is_to_cuda = True
    model_kwargs = configuration.get_pretrained_kwargs()
    assert len(model_kwargs) == 1
    assert model_kwargs["torch_dtype"] == torch.float16 

    transformer = FluxTransformer2DModel.from_pretrained(
        pretrained_diffusion_model_name_or_path,
        subfolder="transformer",
        **model_kwargs,
        )
    #quantize(transformer, weights=qint4, exclude="proj_out")
    quantize(transformer, weights=qint4, exclude="proj_out")
    freeze(transformer)

    transformer.save_pretrained("flux-dev-transformer-float16-qint4")

    qmap = quantization_map(transformer)
    with open("quanto_qmap.json", "w", encoding="utf8") as f:
        json.dump(qmap, f, indent=4)

def test_flux_save_float16_qint4_text_encoder_2():
    test_file_path = test_data_directory / "flux_pipeline_configuration_empty.yml"
    configuration = DiffusionPipelineConfiguration(test_file_path)
    configuration.diffusion_model_path = pretrained_diffusion_model_name_or_path
    configuration.torch_dtype = torch.float16
    configuration.cuda_device = "cuda:0"
    configuration.is_to_cuda = True
    model_kwargs = configuration.get_pretrained_kwargs()
    assert len(model_kwargs) == 1
    assert model_kwargs["torch_dtype"] == torch.float16 

    text_encoder_2 = T5EncoderModel.from_pretrained(
        pretrained_diffusion_model_name_or_path,
        subfolder="text_encoder_2",
        **model_kwargs,
                                                                                                                                                                                                                                                                                )
    quantize(text_encoder_2, weights=qint4)
    freeze(text_encoder_2)
    text_encoder_2.save_pretrained("flux-dev-text-encoder-2-float16-qint4")
    qmap = quantization_map(text_encoder_2)

    with open("quanto_qmap.json", "w", encoding="utf8") as f:
        json.dump(qmap, f, indent=4)

def test_flux_create_pipeline_float16_qint4():
    test_file_path = test_data_directory / "flux_pipeline_configuration_empty.yml"
    configuration = DiffusionPipelineConfiguration(test_file_path)
    configuration.diffusion_model_path = pretrained_diffusion_model_name_or_path
    configuration.torch_dtype = torch.float16
    configuration.cuda_device = "cuda:0"
    configuration.is_to_cuda = True

    path = pretrained_diffusion_model_name_or_path.parents[0] / \
        "flux-dev-transformer-float16-qint4"

    assert path.exists()

    quantized_transformer = QuantizedFluxTransformer2DModel.from_pretrained(
        str(path)).to(
            #device=configuration.cuda_device,
            dtype=configuration.torch_dtype)

    path = pretrained_diffusion_model_name_or_path.parents[0] / \
        "flux-dev-text-encoder-2-float16-qint4"

    assert path.exists()

    quantized_text_encoder_2 = \
        QuantizedT5EncoderModelForCausalLM.from_pretrained(
            str(path)).to(
                #device=configuration.cuda_device,
                dtype=configuration.torch_dtype)

    pipeline = FluxPipeline.from_pretrained(
        pretrained_diffusion_model_name_or_path,
        transformer=transformer,
        text_encoder_2=text_encoder_2,
        torch_dtype=torch.float16)

    #pipeline = create_flux_pipeline(configuration)
    #pipeline.transformer = quantized_transformer
    #pipeline.text_encoder_2 = quantized_text_encoder_2

    #pipeline = change_pipe_to_cuda_or_not(configuration, pipeline)
    pipeline = pipeline.to(configuration.cuda_device)