"""
See
https://github.com/nunchaku-tech/nunchaku/blob/main/examples/v1/sdxl-turbo.py
for the code we're following to develop and test deployment of this nunchaku
model for SDXL Turbo.
"""
from corecode.Utilities import (
    clear_torch_cache_and_collect_garbage,
    DataSubdirectories,
    is_model_there,
    )

from diffusers import StableDiffusionXLPipeline
from nunchaku.models.unets.unet_sdxl import NunchakuSDXLUNet2DConditionModel
from transformers import CLIPTextModel, CLIPTextModelWithProjection

import pytest, torch

data_subdirectories = DataSubdirectories()

relative_nunchaku_model_path = \
    "Models/Diffusion/nunchaku-tech/nunchaku-sdxl-turbo"

is_nunchaku_model_downloaded, nunchaku_model_path = is_model_there(
    relative_nunchaku_model_path,
    data_subdirectories)

relative_sdxl_turbo_model_path = \
    "Models/Diffusion/stabilityai/sdxl-turbo"

is_sdxl_turbo_model_downloaded, sdxl_turbo_model_path = is_model_there(
    relative_sdxl_turbo_model_path,
    data_subdirectories)

@pytest.mark.skipif(
    not is_nunchaku_model_downloaded,
    reason="SDXL Turbo Nunchaku model not there")
def test_NunchakuSDXLUNet2DConditionModel_from_pretrained_works():
    """
    See
    https://github.com/nunchaku-tech/nunchaku/blob/main/examples/v1/sdxl-turbo.py
    for code example that we followed here.
    """
    nunchaku_unet_path = nunchaku_model_path / \
        "svdq-int4_r32-sdxl-turbo.safetensors"
    # class NunchakuSDXLUNet2DConditionModel(UNet2DConditionModel, NunchakuModelLoaderMixin):
    # From
    # https://github.com/nunchaku-tech/nunchaku/blob/main/nunchaku/models/unets/unet_sdxl.py
    # class UNet2DConditionModel(
    # ModelMixin, ConfigMixin, FromOriginalModelMixin, UNet2DConditionLoadersMixin, PeftAdapterMixin
    # ):
    # From diffusers/src/diffusers/models/unets/unet_2d_condition.py
    # def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], **kwargs) -> Self:
    # From diffusers/src/diffusers/models/modeling_utils.py
    unet = NunchakuSDXLUNet2DConditionModel.from_pretrained(
        nunchaku_unet_path,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        # device is keyword from Nunchaku, namely
        # NunchakuSDXLUNet2DConditionModel
        # It appears from the code that value for "device" will overwrite the
        # device configuration.
        # https://github.com/nunchaku-tech/nunchaku/blob/main/nunchaku/models/unets/unet_sdxl.py#L517-L518
        device="cuda:0",
        # device_map is keyword from ModelMixin
        device_map="cuda:0"
        )

    assert unet.sample_size == 64

    assert True

@pytest.mark.skipif(
    not is_nunchaku_model_downloaded or not is_sdxl_turbo_model_downloaded,
    reason="Nunchaku or SDXL Turbo model not there")
def test_StableDiffusionXLPipeline_from_pretrained_works_with_nunchaku_unet():
    """
    See
    https://github.com/nunchaku-tech/nunchaku/blob/main/examples/v1/sdxl-turbo.py
    for code example that we followed here.
    """

    # See
    # diffusers/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl.py
    # class StableDiffusionXLPipeline(
    # DiffusionPipeline,
    # StableDiffusionMixin,
    # FromSingleFileMixin,
    # StableDiffusionXLLoraLoaderMixin,
    # TextualInversionLoaderMixin,
    # IPAdapterMixin,
    # ):
    # DiffusionPipeline's def from_pretrained(..) defines from_pretrained(),
    # diffusers/src/diffusers/pipelines/pipeline_utils.py
    # We obtained this error:
    # E           TypeError: CLIPTextModel.__init__() got an unexpected keyword argument 'offload_state_dict'
    # pipeline = StableDiffusionXLPipeline.from_pretrained(
    #     sdxl_turbo_model_path,
    #     unet=unet,
    #     torch_dtype=torch.bfloat16,
    #     # Load weights from a specified variant filename such as "fp16"
    #     variant="fp16",
    #     local_files_only=True,
    #     ).to("cuda:0")

    # Instead, recall
    #     def __init__(
    #     self,
    #     vae: AutoencoderKL,
    #     text_encoder: CLIPTextModel,
    #     text_encoder_2: CLIPTextModelWithProjection,
    #     tokenizer: CLIPTokenizer,
    #     tokenizer_2: CLIPTokenizer,
    #     unet: UNet2DConditionModel,
    #     scheduler: KarrasDiffusionSchedulers,
    #     image_encoder: CLIPVisionModelWithProjection = None,
    #     feature_extractor: CLIPImageProcessor = None,
    #     force_zeros_for_empty_prompt: bool = True,
    #     add_watermarker: Optional[bool] = None,
    # ):
    # of
    # src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl.py

    text_encoder = CLIPTextModel.from_pretrained(
        pretrained_model_name_or_path=sdxl_turbo_model_path / "text_encoder",
        local_files_only=True,
        device_map="cuda:0",
        )

    # From
    # src/transformers/models/clip/modeling_clip.py
    # class CLIPTextModelWithProjection(CLIPPreTrainedModel):
    # and
    # class CLIPPreTrainedModel(PreTrainedModel):
    # and PreTrainedModel defines def from_pretrained(..) in
    # src/transformers/modeling_utils.py
    # 
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        pretrained_model_name_or_path=sdxl_turbo_model_path / "text_encoder_2",
        local_files_only=True,
        device_map="cuda:0",
        )

    pipeline = StableDiffusionXLPipeline.from_pretrained(
        # See
        # src/diffusers/pipelines/pipeline_utils.py
        pretrained_model_name_or_path=sdxl_turbo_model_path,
        local_files_only=True,
        vae=None,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        unet=None,
        torch_dtype=torch.bfloat16,
        add_watermarker=False,
        device="cuda:0",
        device_map="cuda",
        )

    prompt = (
        "A cinematic shot of a baby racoon wearing an intricate italian priest "
        "robe.")
    negative_prompt = "ugly, blurry, low quality, low resolution"

    # See
    # src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl.py
    # for documentation we followed.

    #         return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

    #     def encode_prompt(
    #     self,
    #     prompt: str,
    #     prompt_2: Optional[str] = None,
    #     device: Optional[torch.device] = None,
    #     num_images_per_prompt: int = 1,
    #     do_classifier_free_guidance: bool = True,
    #     negative_prompt: Optional[str] = None,
    #     negative_prompt_2: Optional[str] = None,
    #     prompt_embeds: Optional[torch.Tensor] = None,
    #     negative_prompt_embeds: Optional[torch.Tensor] = None,
    #     pooled_prompt_embeds: Optional[torch.Tensor] = None,
    #     negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
    #     lora_scale: Optional[float] = None,
    #     clip_skip: Optional[int] = None,
    # ):
    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds) = pipeline.encode_prompt(
            prompt = prompt,
            prompt_2 = prompt,
            negative_prompt = negative_prompt,
            negative_prompt_2 = negative_prompt,
            lora_scale = None,
            clip_skip = None,
        )

    del pipeline.text_encoder
    del pipeline.text_encoder_2
    del pipeline.tokenizer
    del pipeline.tokenizer_2
    del pipeline.scheduler
    del pipeline.image_encoder
    del pipeline.feature_extractor

    del pipeline

    clear_torch_cache_and_collect_garbage()

    nunchaku_unet_path = nunchaku_model_path / \
        "svdq-int4_r32-sdxl-turbo.safetensors"
    unet = NunchakuSDXLUNet2DConditionModel.from_pretrained(
        nunchaku_unet_path,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        device="cuda:0",
        device_map="cuda:0"
        )

    pipeline = StableDiffusionXLPipeline.from_pretrained(
        # See
        # src/diffusers/pipelines/pipeline_utils.py
        pretrained_model_name_or_path=sdxl_turbo_model_path,
        local_files_only=True,
        text_encoder=None,
        text_encoder_2=None,
        tokenizer=None,
        tokenizer_2=None,
        unet=unet,
        torch_dtype=torch.bfloat16,
        add_watermarker=False,
        device="cuda:0",
        device_map="cuda",
    )

    assert pipeline.vae is not None

    image = pipeline(
        # 1024 x 1024 fails because of CUDA out of memory:
        # torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 512.00 MiB. GPU 0 has a total capacity of 7.66 GiB of which 39.81 MiB is free. Process 573436 has 7.60 GiB memory in use. Of the allocated memory 6.97 GiB is allocated by PyTorch, and 456.21 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
        height=512,
        width=512,
        num_inference_steps=30,
        guidance_scale=5.0,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        ).images[0]

    image.save("test_sdxl_turbo_and_nunchaku.png")

    del pipeline.vae
    del pipeline.scheduler
    del pipeline.unet
    del unet
    del pipeline.image_encoder
    del pipeline.feature_extractor

    del pipeline

    clear_torch_cache_and_collect_garbage()

    assert True