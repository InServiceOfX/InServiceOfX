from corecode.Utilities import clear_torch_cache_and_collect_garbage
from nunchaku import NunchakuFluxTransformer2dModel
from nunchaku.lora.flux.compose import compose_lora
from typing import Optional

from morediffusers.Configurations import (
    FluxGenerationConfiguration,
    PipelineInputs,
    NunchakuConfiguration,
    NunchakuLoRAsConfiguration)

from morediffusers.NunchakuWrappers import (
    text_encoder_2_inference,
    )

from morediffusers.Wrappers.pipelines import (
    change_pipe_to_cuda_or_not,
)

from morediffusers.Wrappers import create_seed_generator

from diffusers import FluxKontextPipeline
from diffusers.utils import load_image

class FluxKontextNunchakuAndLoRAs:
    def __init__(
            self,
            configuration: NunchakuConfiguration,
            generation_configuration: FluxGenerationConfiguration,
            pipeline_inputs: PipelineInputs,
            loras_configuration: NunchakuLoRAsConfiguration):
        self._configuration = configuration
        self._generation_configuration = generation_configuration
        self._pipeline_inputs = pipeline_inputs
        self._loras_configuration = loras_configuration

        self._control_images = []

        self._text_encoder_2_enabled = False
        self._transformer_enabled = False

        self._prompt_embeds = []
        self._pooled_prompt_embeds = []
        self._negative_prompt_embeds = []
        self._negative_pooled_prompt_embeds = []
        self._corresponding_prompts = []

    def refresh_configurations(
            self,
            nunchaku_configuration: NunchakuConfiguration,
            flux_generation_configuration: FluxGenerationConfiguration,
            pipeline_inputs: PipelineInputs,
            loras_configuration: NunchakuLoRAsConfiguration):
        self._configuration = nunchaku_configuration
        self._generation_configuration = flux_generation_configuration
        self._pipeline_inputs = pipeline_inputs
        self._loras_configuration = loras_configuration

    @staticmethod
    def _create_corresponding_prompts(
        prompt: str,
        prompt_2: Optional[str],
        negative_prompt: Optional[str],
        negative_prompt_2: Optional[str]):
        prompts = {}

        prompts["prompt"] = prompt
        if prompt_2 is not None:
            prompts["prompt_2"] = prompt_2
        if negative_prompt is not None:
            prompts["negative_prompt"] = negative_prompt
        if negative_prompt_2 is not None:
            prompts["negative_prompt_2"] = negative_prompt_2

        return prompts

    def _create_text_encoder_2_and_pipeline(self):
        if self._text_encoder_2_enabled:
            return

        if self._transformer_enabled:
            self._delete_transformer_and_pipeline()

        path = self._configuration.nunchaku_t5_model_path

        self._text_encoder_2 = \
            text_encoder_2_inference.create_flux_text_encoder_2(
                path,
                self._configuration)

        self._pipeline = \
            text_encoder_2_inference.create_flux_control_text_encoder_2_pipeline(
                self._configuration.flux_model_path,
                self._configuration,
                self._text_encoder_2)

        change_pipe_to_cuda_or_not(self._configuration, self._pipeline)

        self._text_encoder_2_enabled = True

    def _delete_text_encoder_2_and_pipeline(self):
        if not self._text_encoder_2_enabled:
            return

        if hasattr(self._pipeline, "text_encoder"):
            del self._pipeline.text_encoder
        if hasattr(self._pipeline, "text_encoder_2"):
            del self._pipeline.text_encoder_2
        if hasattr(self._pipeline, "tokenizer"):
            del self._pipeline.tokenizer
        if hasattr(self._pipeline, "tokenizer_2"):
            del self._pipeline.tokenizer_2
        del self._text_encoder_2

        clear_torch_cache_and_collect_garbage()

        self._text_encoder_2_enabled = False

    def create_prompt_embeds(self):
        self._create_text_encoder_2_and_pipeline()

        prompt_embeds, pooled_prompt_embeds, text_ids = \
            text_encoder_2_inference.flux_control_encode_prompt(
                self._pipeline,
                self._configuration,
                self._generation_configuration,
                prompt=self._pipeline_inputs.prompt,
                prompt2=self._pipeline_inputs.prompt_2,
                lora_scale=self._loras_configuration.lora_scale)

        negative_prompt_embeds, negative_pooled_prompt_embeds, negative_text_ids = \
            text_encoder_2_inference.flux_control_encode_prompt(
                self._pipeline,
                self._configuration,
                self._generation_configuration,
                prompt=self._pipeline_inputs.negative_prompt,
                prompt2=self._pipeline_inputs.negative_prompt_2,
                lora_scale=self._loras_configuration.lora_scale)

        self._prompt_embeds.append(prompt_embeds)
        self._pooled_prompt_embeds.append(pooled_prompt_embeds)
        self._negative_prompt_embeds.append(negative_prompt_embeds)
        self._negative_pooled_prompt_embeds.append(negative_pooled_prompt_embeds)

        self._corresponding_prompts.append(self._create_corresponding_prompts(
            self._pipeline_inputs.prompt,
            self._pipeline_inputs.prompt_2,
            self._pipeline_inputs.negative_prompt,
            self._pipeline_inputs.negative_prompt_2))

        return (
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
            negative_prompt_embeds,
            negative_pooled_prompt_embeds,
            negative_text_ids)

    def delete_prompt_embeds(self):
        for prompt_embed in self._prompt_embeds:
            del prompt_embed

        for pooled_prompt_embed in self._pooled_prompt_embeds:
            del pooled_prompt_embed

        for negative_prompt_embed in self._negative_prompt_embeds:
            del negative_prompt_embed

        for negative_pooled_prompt_embed in self._negative_pooled_prompt_embeds:
            del negative_pooled_prompt_embed

        del self._prompt_embeds
        del self._pooled_prompt_embeds
        del self._negative_prompt_embeds
        del self._negative_pooled_prompt_embeds
        del self._corresponding_prompts

        self._prompt_embeds = []
        self._pooled_prompt_embeds = []
        self._negative_prompt_embeds = []
        self._negative_pooled_prompt_embeds = []
        self._corresponding_prompts = []

    def load_control_image(self):

        control_image = load_image(str(self._pipeline_inputs.input_image_file_path))
        control_image = control_image.convert("RGB")

        self._control_images.append(control_image)

    def delete_control_images(self):
        for control_image in self._control_images:
            del control_image

        del self._control_images
        self._control_images = []

    def create_transformer_and_pipeline(
            self,
            nunchaku_model_index: Optional[int] = None):
        if nunchaku_model_index is None or \
            nunchaku_model_index < 0 or \
                nunchaku_model_index >= len(
                    self._configuration.nunchaku_model_paths):
            nunchaku_model_index = 0

        if self._transformer_enabled:
            return

        if self._text_encoder_2_enabled:
            self._delete_text_encoder_2_and_pipeline()

        path = self._configuration.nunchaku_model_path[nunchaku_model_index]

        self._transformer = \
            NunchakuFluxTransformer2dModel.from_pretrained(str(path))

        # See class FluxKontextPipeline in pipeline_flux_kontext.py, and
        # notice what it inherits from:
        # class FluxKontextPipeline(
        #     DiffusionPipeline,
        #     FluxLoraLoaderMixin,
        #     FromSingleFileMixin,
        #     TextualInversionLoaderMixin,
        #     FluxIPAdapterMixin,
        # ):
        self._pipeline = FluxKontextPipeline.from_pretrained(
            str(self._configuration.flux_model_path),
            text_encoder=None,
            text_encoder_2=None,
            tokenizer=None,
            tokenizer_2=None,
            transformer=self._transformer,
            torch_dtype=self._configuration.torch_dtype)

        change_pipe_to_cuda_or_not(self._configuration, self._pipeline)

        self._transformer_enabled = True

    def delete_transformer_and_pipeline(self):
        if self._transformer_enabled:
            if hasattr(self._pipeline, "transformer"):
                del self._pipeline.transformer
            del self._pipeline
            del self._transformer
            clear_torch_cache_and_collect_garbage()
            self._transformer_enabled = False

    def update_transformer_with_loras(self):
        if self._transformer_enabled:

            valid_loras = self._loras_configuration.get_valid_loras()

            if len(valid_loras) == 1:
                self._transformer.update_lora_params(str(valid_loras[0][0]))
                self._transformer.set_lora_strength(valid_loras[0][1])
            elif len(valid_loras) > 1:

                loras_to_compose = []
                for valid_lora in valid_loras:
                    loras_to_compose.append(
                        (str(valid_lora[0]), valid_lora[1]))

                composed_loras = compose_lora(loras_to_compose)
                self._transformer.update_lora_params(composed_loras)

    def call_pipeline(self, prompt_embed_index, control_image_index):
        if (not self._prompt_embeds) or \
            (not self._pooled_prompt_embeds) or \
            (not self._negative_prompt_embeds) or \
            (not self._negative_pooled_prompt_embeds) or \
            (not self._control_images):
            return None

        if (prompt_embed_index < 0 or \
            prompt_embed_index >= len(self._prompt_embeds)) or \
            (prompt_embed_index < 0 or \
                prompt_embed_index >= len(self._pooled_prompt_embeds)) or \
            (control_image_index < 0 or \
                control_image_index >= len(self._control_images)):
            return None

        true_cfg_scale = self._generation_configuration.true_cfg_scale
        if true_cfg_scale is None:
            true_cfg_scale = 1.0

        # See class FluxKontextPipeline in pipeline_flux_kontext.py, and
        # def __call__(..).
        return self._pipeline(
            image=self._control_images[control_image_index],
            prompt_embeds=self._prompt_embeds[prompt_embed_index],
            pooled_prompt_embeds=self._pooled_prompt_embeds[prompt_embed_index],
            negative_prompt_embeds=self._negative_prompt_embeds[
                prompt_embed_index],
            negative_pooled_prompt_embeds=self._negative_pooled_prompt_embeds[
                prompt_embed_index],
            true_cfg_scale=true_cfg_scale,
            height=self._generation_configuration.height,
            width=self._generation_configuration.width,
            num_inference_steps=\
                self._generation_configuration.num_inference_steps,
            guidance_scale=self._generation_configuration.guidance_scale,
            generator=create_seed_generator(
                self._configuration,
                self._generation_configuration)).images

    def restart(self):
        self._delete_text_encoder_2_and_pipeline()
        self.delete_prompt_embeds()
        self.delete_control_images()
        self.delete_transformer_and_pipeline()
