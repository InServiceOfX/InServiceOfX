from corecode.Utilities import clear_torch_cache_and_collect_garbage
from nunchaku import NunchakuFluxTransformer2dModel
from typing import Optional

from morediffusers.Configurations import (
    FluxGenerationConfiguration,
    PipelineInputs,
    NunchakuConfiguration,
    NunchakuLoRAsConfiguration)

from morediffusers.NunchakuWrappers import (
    text_encoder_2_inference,
    transformer_inference,
    )

from morediffusers.Wrappers.pipelines import (
    change_pipe_to_cuda_or_not,
)

class FluxNunchakuAndLoRAs:
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

    def is_text_encoder_2_enabled(self):
        return self._text_encoder_2_enabled

    def is_transformer_enabled(self):
        return self._transformer_enabled

    def _create_text_encoder_2_and_pipeline(self):
        if self._text_encoder_2_enabled:
            return

        path = self._configuration.nunchaku_t5_model_path

        self._text_encoder_2 = \
            text_encoder_2_inference.create_flux_text_encoder_2(
                path,
                self._configuration)

        self._pipeline = \
            text_encoder_2_inference.create_flux_text_encoder_2_pipeline(
                self._configuration.flux_model_path,
                self._configuration,
                self._text_encoder_2)

        change_pipe_to_cuda_or_not(self._configuration, self._pipeline)

        self._text_encoder_2_enabled = True

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

    def create_prompt_embeds(self):

        self._create_text_encoder_2_and_pipeline()

        prompt_embeds, pooled_prompt_embeds, text_ids = \
            text_encoder_2_inference.encode_prompt(
                pipeline=self._pipeline,
                generation_configuration=self._generation_configuration,
                prompt=self._pipeline_inputs.prompt,
                prompt2=self._pipeline_inputs.prompt_2,
                device=self._configuration.cuda_device,
                lora_scale=self._loras_configuration.lora_scale)

        negative_prompt_embeds, negative_pooled_prompt_embeds, negative_text_ids = \
            text_encoder_2_inference.encode_prompt(
                pipeline=self._pipeline,
                generation_configuration=self._generation_configuration,
                prompt=self._pipeline_inputs.negative_prompt,
                prompt2=self._pipeline_inputs.negative_prompt_2,
                device=self._configuration.cuda_device,
                lora_scale=self._loras_configuration.lora_scale)

        self._prompt_embeds.append(prompt_embeds)
        self._pooled_prompt_embeds.append(pooled_prompt_embeds)
        self._negative_prompt_embeds.append(negative_prompt_embeds)
        self._negative_pooled_prompt_embeds.append(
            negative_pooled_prompt_embeds)

        self._corresponding_prompts = self._create_corresponding_prompts(
            self._pipeline_inputs.prompt,
            self._pipeline_inputs.prompt_2,
            self._pipeline_inputs.negative_prompt,
            self._pipeline_inputs.negative_prompt_2)

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
        self._corresponding_prompts = None

    def delete_text_encoder_2_and_pipeline(self):
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

    def create_transformer_and_pipeline(self):
        if self._transformer_enabled:
            return

        if self._text_encoder_2_enabled:
            self.delete_text_encoder_2_and_pipeline()
            self._text_encoder_2_enabled = False

        path = self._configuration.nunchaku_model_path

        self._transformer = \
            NunchakuFluxTransformer2dModel.from_pretrained(str(path))

        self._pipeline = transformer_inference.create_flux_transformer_pipeline(
            str(self._configuration.flux_model_path),
            self._configuration,
            self._transformer)

        change_pipe_to_cuda_or_not(self._configuration, self._pipeline)

        self._transformer_enabled = True

    def delete_transformer_and_pipeline(self):
        del self._pipeline.transformer
        del self._pipeline.transformer_2
        del self._pipeline.tokenizer
        del self._pipeline.tokenizer_2
        del self._pipeline
        del self._transformer

        clear_torch_cache_and_collect_garbage()

        self._transformer_enabled = False

    def update_transformer_with_lora(self, valid_lora):
        if self._transformer_enabled:
            self._transformer.update_lora_params(str(valid_lora[0]))
            self._transformer.set_lora_strength(valid_lora[1])

    def update_transformer_with_loras(self):
        if self._transformer_enabled:

            valid_loras = self._loras_configuration.get_valid_loras()

            for lora in valid_loras:
                self._transformer.update_lora_params(str(lora[0]))
                self._transformer.set_lora_strength(lora[1])

    def call_pipeline(
        self,
        prompt_embeds,
        pooled_prompt_embeds,
        negative_prompt_embeds,
        negative_pooled_prompt_embeds):
        if self._transformer_enabled:
            return transformer_inference.call_pipeline(
                self._pipeline,
                prompt_embeds,
                pooled_prompt_embeds,
                self._configuration,
                self._generation_configuration,
                negative_prompt_embeds,
                negative_pooled_prompt_embeds).images

    def call_pipeline_with_prompt_embed(self, index):
        if (not self._prompt_embeds) or \
            (not self._pooled_prompt_embeds) or \
            (not self._negative_prompt_embeds) or \
            (not self._negative_pooled_prompt_embeds):
            return None

        if (index < 0 or index >= len(self._prompt_embeds)) or \
            (index < 0 or index >= len(self._pooled_prompt_embeds)) or \
            (index < 0 or index >= len(self._negative_prompt_embeds)) or \
            (index < 0 or index >= len(self._negative_pooled_prompt_embeds)):
            return None

        return self.call_pipeline(
            self._prompt_embeds[index],
            self._pooled_prompt_embeds[index],
            self._negative_prompt_embeds[index],
            self._negative_pooled_prompt_embeds[index])