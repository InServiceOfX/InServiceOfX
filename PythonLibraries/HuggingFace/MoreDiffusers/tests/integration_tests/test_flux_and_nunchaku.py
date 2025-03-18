from corecode.Utilities import (
    clear_torch_cache_and_collect_garbage,
    DataSubdirectories,
    )

from morediffusers.Configurations import DiffusionPipelineConfiguration
from morediffusers.Wrappers.pipelines import (
    change_pipe_to_cuda_or_not,
    )

from diffusers import FluxPipeline

from pathlib import Path
import torch

# Instructions from README.md of
# https://huggingface.co/mit-han-lab/svdq-int4-flux.1-dev
# look to be old and wrong. Using README.md from here:
# https://github.com/mit-han-lab/nunchaku

from nunchaku import NunchakuFluxTransformer2dModel, NunchakuT5EncoderModel

data_sub_dirs = DataSubdirectories()
test_data_directory = Path(__file__).resolve().parents[1] / "TestData"

pretrained_diffusion_model_name_or_path = \
    data_sub_dirs.ModelsDiffusion / "black-forest-labs" / "FLUX.1-dev"
if not pretrained_diffusion_model_name_or_path.exists():
    pretrained_diffusion_model_name_or_path = \
        Path("/Data1/Models/Diffusion/") / "black-forest-labs" / "FLUX.1-dev"

def test_flux_dev_and_nunchaku():
    """
    Instructions from README.md of
    https://huggingface.co/mit-han-lab/svdq-int4-flux.1-dev
    look to be old and wrong. Using README.md from here:
    https://github.com/mit-han-lab/nunchaku
    """
    path = pretrained_diffusion_model_name_or_path.parents[1] / \
        "mit-han-lab" / "svdq-int4-flux.1-dev"
    assert path.exists()
    transformer = NunchakuFluxTransformer2dModel.from_pretrained(path)

    path = pretrained_diffusion_model_name_or_path.parents[1] / \
        "mit-han-lab" / "svdq-flux.1-t5"
    text_encoder_2 = NunchakuT5EncoderModel.from_pretrained(path)

    pipeline = FluxPipeline.from_pretrained(
        pretrained_diffusion_model_name_or_path,
        text_encoder_2=text_encoder_2,
        transformer=transformer,
        torch_dtype=torch.bfloat16,)

    test_file_path = test_data_directory / "flux_pipeline_configuration_empty.yml"
    configuration = DiffusionPipelineConfiguration(test_file_path)
    configuration.torch_dtype = torch.bfloat16
    configuration.cuda_device = "cuda:0"
    configuration.is_to_cuda = True
    # Out of Memory when without text_encoder_2
    #pipeline = change_pipe_to_cuda_or_not(configuration, pipeline)

    # Out of Memory when without text_encoder_2
    pipeline.to("cuda:0")

import gc

def test_flux_dev_nunchaku_and_only_text_encoder_2():
    """
    https://huggingface.co/docs/diffusers/en/training/distributed_inference
    See Model sharding - distributes models across GPUs when models don't fit on
    a single GPU.
    https://huggingface.co/docs/diffusers/en/training/distributed_inference#model-sharding
    """

    prompt = "a photo of a dog with cat-like look"

    path = pretrained_diffusion_model_name_or_path.parents[1] / \
        "mit-han-lab" / "svdq-flux.1-t5"
    text_encoder_2 = NunchakuT5EncoderModel.from_pretrained(path)

    # device_map="cuda:0" does not work,
    # NotImplementedError: cuda:0 not supported. Supported strategies are: balanced
    pipeline = FluxPipeline.from_pretrained(
        pretrained_diffusion_model_name_or_path,
        text_encoder_2=text_encoder_2,
        transformer=None,
        torch_dtype=torch.bfloat16,)
        # If you have device_map="balanced" with running .to(..) you get this:
        # ValueError: It seems like you have activated a device mapping strategy
        # on the pipeline which doesn't allow explicit device placement using `to()`.
        # You can call `reset_device_map()` to remove the existing device map
        # from the pipeline.
#        device_map="balanced")

    pipeline.to(device="cuda:0", dtype=torch.bfloat16)

    with torch.no_grad():
        print("Encoding prompts.")
        prompt_embeds, pooled_prompt_embeds, text_ids = pipeline.encode_prompt(
            prompt=prompt, prompt_2=None, max_sequence_length=512)

    def flush():
        gc.collect()
        torch.cuda.empty_cache()
        # Uncomment the following if max memory had been set before.
        #torch.cuda.reset_max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()

    del pipeline.text_encoder
    del pipeline.text_encoder_2
    del pipeline.tokenizer
    del pipeline.tokenizer_2
    flush()

def test_flux_dev_nunchaku_transformer_and_vae():
    """
    Fails on 3060. Out of memory.

    See Model sharding - distributes models across GPUs when models don't fit on
    a single GPU.
    https://huggingface.co/docs/diffusers/en/training/distributed_inference#model-sharding
    """

    prompt = "a photo of a dog with cat-like look"

    path = pretrained_diffusion_model_name_or_path.parents[1] / \
        "mit-han-lab" / "svdq-flux.1-t5"
    text_encoder_2 = NunchakuT5EncoderModel.from_pretrained(path)

    # device_map="cuda:0" does not work,
    # NotImplementedError: cuda:0 not supported. Supported strategies are: balanced
    pipeline = FluxPipeline.from_pretrained(
        pretrained_diffusion_model_name_or_path,
        text_encoder_2=text_encoder_2,
        transformer=None,
        torch_dtype=torch.bfloat16,)
        # If you have device_map="balanced" with running .to(..) you get this:
        # ValueError: It seems like you have activated a device mapping strategy
        # on the pipeline which doesn't allow explicit device placement using `to()`.
        # You can call `reset_device_map()` to remove the existing device map
        # from the pipeline.
#        device_map="balanced")

    pipeline.to(device="cuda:0", dtype=torch.bfloat16)

    with torch.no_grad():
        print("Encoding prompts.")
        prompt_embeds, pooled_prompt_embeds, text_ids = pipeline.encode_prompt(
            prompt=prompt, prompt_2=None, max_sequence_length=512)

    def flush():
        gc.collect()
        torch.cuda.empty_cache()
        # Uncomment the following if max memory had been set before.
        #torch.cuda.reset_max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.ipc_collect()

    del pipeline.text_encoder
    del pipeline.text_encoder_2
    del pipeline.tokenizer
    del pipeline.tokenizer_2
    flush()

    path = pretrained_diffusion_model_name_or_path.parents[1] / \
        "mit-han-lab" / "svdq-int4-flux.1-dev"
    assert path.exists()

    # Out of memory on 3060 only:
    # 
    transformer = NunchakuFluxTransformer2dModel.from_pretrained(path)

    pipeline = FluxPipeline.from_pretrained(
        pretrained_diffusion_model_name_or_path,
        text_encoder=None,
        text_encoder_2=None,
        tokenizer=None,
        tokenizer_2=None,
        transformer=transformer,
        torch_dtype=torch.bfloat16)

    pipeline.to(device="cuda:0", dtype=torch.bfloat16)

    height, width = 768, 1360
    image = pipeline(
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        num_inference_steps=50,
        guidance_scale=3.5,
        height=height,
        width=width,).images[0]

    save_path = Path.cwd() / "test_flux_nunchaku_sharding.png"
    image.save(str(save_path))

    del pipeline.transformer
    del pipeline
    flush()

def test_flux_dev_nunchaku_text_encoder_2_transformer_and_vae():
    """
    This works on single 3060.

    See Model sharding - distributes models across GPUs when models don't fit on
    a single GPU.
    https://huggingface.co/docs/diffusers/en/training/distributed_inference#model-sharding
    """

    prompt = "a photo of a dog with cat-like look"

    path = pretrained_diffusion_model_name_or_path.parents[1] / \
        "mit-han-lab" / "svdq-flux.1-t5"
    text_encoder_2 = NunchakuT5EncoderModel.from_pretrained(path)

    # device_map="cuda:0" does not work,
    # NotImplementedError: cuda:0 not supported. Supported strategies are: balanced
    pipeline = FluxPipeline.from_pretrained(
        pretrained_diffusion_model_name_or_path,
        text_encoder_2=text_encoder_2,
        transformer=None,
        vae=None,
        torch_dtype=torch.bfloat16,)

    pipeline.to(device="cuda:0", dtype=torch.bfloat16)

    with torch.no_grad():
        print("Encoding prompts.")
        prompt_embeds, pooled_prompt_embeds, text_ids = pipeline.encode_prompt(
            prompt=prompt, prompt_2=None, max_sequence_length=512)

    def flush():
        gc.collect()
        torch.cuda.empty_cache()
        # Uncomment the following if max memory had been set before.
        #torch.cuda.reset_max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.ipc_collect()

    del pipeline.text_encoder
    del pipeline.text_encoder_2
    del pipeline.tokenizer
    del pipeline.tokenizer_2
    # Empirically, I saw VRAM usage drop significantly after I deleted the
    # variable text_encoder_2. I think this is because *all* variable references
    # must be deleted for the GPU VRAM to be freed.
    del text_encoder_2

    flush()

    path = pretrained_diffusion_model_name_or_path.parents[1] / \
        "mit-han-lab" / "svdq-int4-flux.1-dev"
    assert path.exists()

    transformer = NunchakuFluxTransformer2dModel.from_pretrained(path)

    pipeline = FluxPipeline.from_pretrained(
        pretrained_diffusion_model_name_or_path,
        text_encoder=None,
        text_encoder_2=None,
        tokenizer=None,
        tokenizer_2=None,
        transformer=transformer,
        torch_dtype=torch.bfloat16)

    pipeline.to(device="cuda:0", dtype=torch.bfloat16)

    height, width = 768, 1360
    image = pipeline(
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        num_inference_steps=50,
        guidance_scale=3.5,
        height=height,
        width=width,).images[0]

    save_path = Path.cwd() / "test_flux_nunchaku_sharding.png"
    image.save(str(save_path))

    del pipeline.transformer
    del pipeline
    del transformer
    flush()


def test_flux_dev_nunchaku_transformer_vae_multigpu():
    """
    Fails for NVIDIA GeForce RTX 3060 and 980 Ti
    """

    prompt = "a photo of a dog with cat-like look"

    path = pretrained_diffusion_model_name_or_path.parents[1] / \
        "mit-han-lab" / "svdq-flux.1-t5"
    text_encoder_2 = NunchakuT5EncoderModel.from_pretrained(path)

    # device_map="cuda:0" does not work,
    # NotImplementedError: cuda:0 not supported. Supported strategies are: balanced
    pipeline = FluxPipeline.from_pretrained(
        pretrained_diffusion_model_name_or_path,
        text_encoder_2=text_encoder_2,
        transformer=None,
        torch_dtype=torch.bfloat16,
        device_map="balanced",
        max_memory={0: "4GB", 1: "11GB"})

    # Commented out because it doesn't work:
    # ValueError: It seems like you have activated a device mapping strategy on
    # the pipeline which doesn't allow explicit device placement using `to()`.
    # You can call `reset_device_map()` to remove the existing device map from
    # the pipeline.
    #pipeline.to(device="cuda", dtype=torch.bfloat16)

    device = torch.device("cuda")

    with torch.no_grad():
        print("Encoding prompts.")
        prompt_embeds, pooled_prompt_embeds, text_ids = pipeline.encode_prompt(
            prompt=prompt,
            prompt_2=None,
            max_sequence_length=512,
            device=device)

    def flush():
        gc.collect()
        torch.cuda.empty_cache()
        # Uncomment the following if max memory had been set before.
        #torch.cuda.reset_max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.ipc_collect()

    del pipeline.text_encoder
    del pipeline.text_encoder_2
    del pipeline.tokenizer
    del pipeline.tokenizer_2
    flush()

    path = pretrained_diffusion_model_name_or_path.parents[1] / \
        "mit-han-lab" / "svdq-int4-flux.1-dev"
    assert path.exists()
    transformer = NunchakuFluxTransformer2dModel.from_pretrained(path)

    pipeline = FluxPipeline.from_pretrained(
        pretrained_diffusion_model_name_or_path,
        text_encoder=None,
        text_encoder_2=None,
        tokenizer=None,
        tokenizer_2=None,
        transformer=transformer,
        torch_dtype=torch.bfloat16)

    pipeline.to(device="cuda:0", dtype=torch.bfloat16)

    height, width = 768, 1360
    image = pipeline(
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        num_inference_steps=50,
        guidance_scale=3.5,
        height=height,
        width=width,).images[0]

    save_path = Path.cwd() / "test_flux_nunchaku_sharding_multigpu.png"
    image.save(str(save_path))

    del pipeline.transformer
    del pipeline
    flush()

from morediffusers.Configurations import (
    DiffusionPipelineConfiguration,
    FluxGenerationConfiguration
)

from morediffusers.Wrappers import create_seed_generator

def test_flux_dev_nunchaku_and_configurations():
    """
    This works on a single NVIDIA GeForce RTX 3060.
    """

    test_file_path = test_data_directory / "flux_pipeline_configuration_empty.yml"
    configuration = DiffusionPipelineConfiguration(test_file_path)
    configuration.cuda_device = "cuda:0"
    configuration.torch_dtype = torch.bfloat16

    # https://civitai.com/images/25627842
    prompt = (
        "A captivating, high-resolution portrait-style image of a woman emerging"
        " from the water at sunset. The composition is centered, emphasizing her"
        " face and upper torso. Her wet, slicked-back hair enhances her striking"
        " features, with high cheekbones, full lips, and piercing blue eyes. She"
        " wears a delicate gold necklace and small, intricate gold earrings, "
        "accentuating her natural beauty. The background showcases a serene "
        "ocean and a stunning gradient sky, transitioning from deep blue to warm "
        "orange hues, signifying sunset. Water droplets on her skin add a "
        "refreshing touch to the artwork. The artist, Meg B., masterfully "
        "captures the essence of freshness and natural beauty in this vivid, "
        "conceptual piece., 3d render, vibrant, photo, conceptual art, "
        "illustration, poster, painting")

    path = pretrained_diffusion_model_name_or_path.parents[1] / \
        "mit-han-lab" / "svdq-flux.1-t5"
    text_encoder_2 = NunchakuT5EncoderModel.from_pretrained(
        path,
        # device_map=configuration.cuda_device does not work because when a
        # NunchakuFluxTransformer2dModel is created, out of memory error.
        #device_map=configuration.cuda_device,
        torch_dtype=configuration.torch_dtype)

    # NunchakuT5EncoderModel, which derives from T5EncoderModel from
    # huggingface's transformers, does not have the enable_model_cpu_offload
    # method.
    #text_encoder_2.enable_model_cpu_offload(**kwargs)

    pipeline = FluxPipeline.from_pretrained(
        pretrained_diffusion_model_name_or_path,
        text_encoder_2=text_encoder_2,
        transformer=None,
        vae=None,
        torch_dtype=configuration.torch_dtype)

    pipeline.to(
        device=configuration.cuda_device,
        dtype=configuration.torch_dtype)

    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds, text_ids = pipeline.encode_prompt(
            prompt=prompt,
            prompt_2=None,
            max_sequence_length=512)

    del pipeline.text_encoder
    del pipeline.text_encoder_2
    del pipeline.tokenizer
    del pipeline.tokenizer_2
    del text_encoder_2

    clear_torch_cache_and_collect_garbage()

    path = pretrained_diffusion_model_name_or_path.parents[1] / \
        "mit-han-lab" / "svdq-int4-flux.1-dev"
    assert path.exists()

    transformer = NunchakuFluxTransformer2dModel.from_pretrained(path)

    pipeline = FluxPipeline.from_pretrained(
        pretrained_diffusion_model_name_or_path,
        text_encoder=None,
        text_encoder_2=None,
        tokenizer=None,
        tokenizer_2=None,
        transformer=transformer,
        torch_dtype=configuration.torch_dtype)

    pipeline.to(device="cuda:0", dtype=torch.bfloat16)

    test_file_path = test_data_directory / "flux_generation_configuration_empty.yml"
    generation_configuration = FluxGenerationConfiguration(test_file_path)

    generation_configuration.height = 1216
    generation_configuration.width = 832
    generation_configuration.num_inference_steps = 40
    generation_configuration.seed = 2788402957
    generation_configuration.guidance_scale = 3.5

    generated_seed = create_seed_generator(
        configuration,
        generation_configuration)

    image = pipeline(
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        num_inference_steps=generation_configuration.num_inference_steps,
        guidance_scale=generation_configuration.guidance_scale,
        height=generation_configuration.height,
        width=generation_configuration.width,
        generator=generated_seed).images[0]

    save_path = Path.cwd() / "test_flux_nunchaku_and_configurations.png"
    image.save(str(save_path))

    del pipeline.transformer
    del pipeline
    del transformer
    clear_torch_cache_and_collect_garbage()

from morediffusers.Configurations import FluxGenerationConfiguration
from morediffusers.NunchakuWrappers import (
    text_encoder_2_inference,
    transformer_inference,
    )

def test_nunchaku_wrappers():
    test_file_path = test_data_directory / "flux_pipeline_configuration_empty.yml"
    configuration = DiffusionPipelineConfiguration(test_file_path)
    configuration.cuda_device = "cuda:0"
    configuration.torch_dtype = torch.bfloat16
    configuration.is_to_cuda = True

    # https://civitai.com/images/22934060
    prompt = (
        "30yo American woman wearing bussiness suit holding a cigarrete "
        "Dark black smooth long hair ponytail. smiling. Looking at camera. "
        "Dynamic varying pose. Outside business. Night time, Night time, dark.")

    path = pretrained_diffusion_model_name_or_path.parents[1] / \
        "mit-han-lab" / "svdq-flux.1-t5"

    text_encoder_2 = text_encoder_2_inference.create_flux_text_encoder_2(
        path,
        configuration)

    pipeline = text_encoder_2_inference.create_flux_text_encoder_2_pipeline(
        pretrained_diffusion_model_name_or_path,
        configuration,
        text_encoder_2)

    change_pipe_to_cuda_or_not(configuration, pipeline)

    generation_configuration = FluxGenerationConfiguration()

    generation_configuration.height = 1216
    generation_configuration.width = 832
    generation_configuration.num_inference_steps = 34
    generation_configuration.seed = 1544597505
    generation_configuration.guidance_scale = 1.5
    generation_configuration.max_sequence_length = 512

    prompt_embeds, pooled_prompt_embeds, text_ids = \
        text_encoder_2_inference.encode_prompt(
            pipeline,
            generation_configuration,
            prompt)

    # text_encoder_2_inference.delete_variables_on_device(
    #     pipeline,
    #     text_encoder_2)

    del pipeline.text_encoder
    del pipeline.text_encoder_2
    del pipeline.tokenizer
    del pipeline.tokenizer_2
    del text_encoder_2

    clear_torch_cache_and_collect_garbage()

    path = pretrained_diffusion_model_name_or_path.parents[1] / \
        "mit-han-lab" / "svdq-int4-flux.1-dev"
    assert path.exists()

    transformer = transformer_inference.create_flux_transformer(path)

    pipeline = transformer_inference.create_flux_transformer_pipeline(
        pretrained_diffusion_model_name_or_path,
        configuration,
        transformer)

    change_pipe_to_cuda_or_not(configuration, pipeline)

    image = transformer_inference.call_pipeline(
        pipeline,
        prompt_embeds,
        pooled_prompt_embeds,
        configuration,
        generation_configuration).images[0]

    save_path = Path.cwd() / "test_nunchaku_wrappers.png"
    image.save(str(save_path))

    transformer_inference.delete_variables_on_device(
        pipeline,
        transformer)

    clear_torch_cache_and_collect_garbage()

def test_nunchaku_svdq_lora():
    test_file_path = test_data_directory / "flux_pipeline_configuration_empty.yml"
    configuration = DiffusionPipelineConfiguration(test_file_path)
    configuration.cuda_device = "cuda:0"
    configuration.torch_dtype = torch.bfloat16
    configuration.is_to_cuda = True

    prompt=(
        "GHIBSKY style, cozy mountain cabin covered in snow, with smoke curling"
        " from the chimney and a warm, inviting light spilling through the "
        "windows")

    path = pretrained_diffusion_model_name_or_path.parents[1] / \
        "mit-han-lab" / "svdq-flux.1-t5"

    text_encoder_2 = text_encoder_2_inference.create_flux_text_encoder_2(
        path,
        configuration)

    pipeline = text_encoder_2_inference.create_flux_text_encoder_2_pipeline(
        pretrained_diffusion_model_name_or_path,
        configuration,
        text_encoder_2)

    change_pipe_to_cuda_or_not(configuration, pipeline)

    generation_configuration = FluxGenerationConfiguration()

    generation_configuration.height = 1216
    generation_configuration.width = 832
    generation_configuration.num_inference_steps = 34
    generation_configuration.seed = 1544597505
    generation_configuration.guidance_scale = 1.5
    generation_configuration.max_sequence_length = 512

    prompt_embeds, pooled_prompt_embeds, text_ids = \
        text_encoder_2_inference.encode_prompt(
            pipeline,
            generation_configuration,
            prompt)

    del pipeline.text_encoder
    del pipeline.text_encoder_2
    del pipeline.tokenizer
    del pipeline.tokenizer_2
    del text_encoder_2

    clear_torch_cache_and_collect_garbage()

    path = pretrained_diffusion_model_name_or_path.parents[1] / \
        "mit-han-lab" / "svdq-int4-flux.1-dev"
    assert path.exists()

    transformer = transformer_inference.create_flux_transformer(path)

    pipeline = transformer_inference.create_flux_transformer_pipeline(
        pretrained_diffusion_model_name_or_path,
        configuration,
        transformer)

    change_pipe_to_cuda_or_not(configuration, pipeline)

    filepath = pretrained_diffusion_model_name_or_path.parents[1] / \
        "mit-han-lab" / "svdquant-lora-collection" / \
        "svdq-int4-flux.1-dev-realism.safetensors"
    assert filepath.exists()

    transformer.update_lora_params(str(filepath))
    transformer.set_lora_strength(1)

    image = transformer_inference.call_pipeline(
        pipeline,
        prompt_embeds,
        pooled_prompt_embeds,
        configuration,
        generation_configuration).images[0]

    save_path = Path.cwd() / "test_nunchaku_svdq_lora.png"
    image.save(str(save_path))

    transformer_inference.delete_variables_on_device(
        pipeline,
        transformer)

    clear_torch_cache_and_collect_garbage()
