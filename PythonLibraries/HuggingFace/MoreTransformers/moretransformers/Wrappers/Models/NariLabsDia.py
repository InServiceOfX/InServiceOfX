from moretransformers.Configurations import GenerationConfiguration
from moretransformers.Conversions import convert_mp3_to_AudioInput
from transformers import DiaProcessor, DiaForConditionalGeneration

from typing import Optional
from pathlib import Path

class NariLabsDia:
    @staticmethod
    def create_default_generation_configuration() -> GenerationConfiguration:
        """
        These values were suggested by Nari Labs.
        """
        return GenerationConfiguration(
            max_new_tokens=3072,
            temperature=1.8,
            top_p=0.90,
            top_k=50,
        )

    def __init__(
            self,
            model_path: str | Path,
            generation_configuration: GenerationConfiguration,
            sampling_rate: int = 44100,
            device_map: str = "cuda:0",
            guidance_scale: Optional[float] = None):
        self._device_map = device_map

        if guidance_scale is None:
            # From Nari Lab's suggestion.
            guidance_scale = 3.0

        self._sampling_rate = sampling_rate
        self._setup_processor_and_model(model_path, device_map)
        self._setup_generation(generation_configuration, guidance_scale)

    def _setup_generation(
            self,
            generation_configuration,
            guidance_scale: float):
        self._generation_configuration = generation_configuration
        self._guidance_scale = guidance_scale

    def _setup_processor_and_model(
            self,
            model_path: str | Path,
            device_map: str = "cuda:0"):
        self._processor = DiaProcessor.from_pretrained(model_path)
        self._model = DiaForConditionalGeneration.from_pretrained(
            model_path,
            local_files_only=True).to(device_map)

    def process_text_only(self, text: str | list[str]):
        return self._processor(
            text=text,
            padding=True,
            return_tensors="pt",
            sampling_rate=self._sampling_rate).to(self._device_map)

    def _process_text_and_audio(
            self,
            text: str | list[str],
            audio):

        if isinstance(text, str) or (
            isinstance(text, list) and len(text) == 1):
            inputs = self._processor(
                text=text,
                audio=audio,
                padding=True,
                return_tensors="pt").to(self._device_map)
        elif isinstance(text, list) and len(text) > 1:
            inputs = self._processor(
                text=text,
                audio=[audio] * len(text),
                padding=True,
                return_tensors="pt").to(self._device_map)
        else:
            raise ValueError(f"Invalid text: {text}")

        prompt_len = \
            self._processor.get_audio_prompt_len(
                inputs["decoder_attention_mask"])

        return inputs, prompt_len

    def process_text_and_mp3(
            self,
            text: str | list[str],
            audio_file_path: str | Path):
        audio, _ = convert_mp3_to_AudioInput(audio_file_path)
        return self._process_text_and_audio(text, audio)

    def generate_from_text_only(self, inputs, output_path: str | Path):
        outputs = self._model.generate(
            **inputs,
            max_new_tokens=self._generation_configuration.max_new_tokens,
            guidance_scale=self._guidance_scale,
            temperature=self._generation_configuration.temperature,
            top_p=self._generation_configuration.top_p,
            top_k=self._generation_configuration.top_k)

        outputs = self._processor.batch_decode(outputs)
        self._processor.save_audio(outputs, str(output_path))
        return outputs

    def generate_from_text_and_audio(
            self,
            inputs,
            output_path: str | Path,
            audio_prompt_len: int):
        outputs = self._model.generate(
            **inputs,
            max_new_tokens=self._generation_configuration.max_new_tokens,
            guidance_scale=self._guidance_scale,
            temperature=self._generation_configuration.temperature,
            top_p=self._generation_configuration.top_p,
            top_k=self._generation_configuration.top_k)

        outputs = self._processor.batch_decode(
            outputs,
            audio_prompt_len=audio_prompt_len)
        self._processor.save_audio(outputs, str(output_path))
        return outputs

    def generate_batch_from_text_and_audio(
            self,
            inputs,
            output_path: str | Path,
            output_file_name_prefix: str,
            audio_prompt_len: int):
        outputs = self._model.generate(
            **inputs,
            max_new_tokens=self._generation_configuration.max_new_tokens,
            guidance_scale=self._guidance_scale,
            temperature=self._generation_configuration.temperature,
            top_p=self._generation_configuration.top_p,
            top_k=self._generation_configuration.top_k)

        outputs = self._processor.batch_decode(
            outputs,
            audio_prompt_len=audio_prompt_len)

        output_path = Path(output_path)

        saving_paths = []
        for i in range(len(outputs)):
            saving_paths.append(
                output_path / f"{output_file_name_prefix}_{i}.mp3")
        saving_paths = [str(path) for path in saving_paths]

        self._processor.save_audio(outputs, saving_paths)
        return outputs