from corecode.FileIO import TextFile

from moretransformers.Configurations import (
    FromPretrainedModelConfiguration,
)

from moretransformers.Configurations.TextToSpeech import (
    VibeVoiceConfiguration,
)

from transformers.models.vibevoice.vibevoice_processor import (
    VibeVoiceProcessor,
)
from transformers.models.vibevoice.modeling_vibevoice_inference import (
    VibeVoiceForConditionalGenerationInference,
)

from pathlib import Path
from typing import List
import torch

class VibeVoiceModelAndProcessor:
    def __init__(
            self,
            from_pretrained_model_configuration: \
                FromPretrainedModelConfiguration,
            vibe_voice_configuration: VibeVoiceConfiguration,
    ):
        self.fpm_configuration = from_pretrained_model_configuration
        self.vv_configuration = vibe_voice_configuration

        if self.fpm_configuration.pretrained_model_name_or_path is None:
            raise ValueError(
                "VibeVoiceModelAndProcessor.__init__ pretrained_model_name_or_path is required")

        self._inputs = None

    def _text_files_to_strings(self):
        text_file_paths = self.vv_configuration.text_file_paths
        text_strings = []
        for text_file_path in text_file_paths:
            text_strings.append(TextFile.load_text(text_file_path))
        return text_strings

    def load_processor(self):
        self._processor = VibeVoiceProcessor.from_pretrained(
            pretrained_model_name_or_path=\
                self.fpm_configuration.pretrained_model_name_or_path)

    def process_inputs(self, texts: List[str] = None):
        if texts is None:
            texts = self._text_files_to_strings()

        voice_sample_paths = [
            str(audio_file_path) for audio_file_path in \
                self.vv_configuration.audio_file_paths
        ]

        self._inputs = self._processor(
            text=texts,
            voice_samples=voice_sample_paths,
            return_tensors="pt",
            padding=True,
        )
        if self.fpm_configuration.device_map is not None:
            self._inputs = {
                k: v.to(self.fpm_configuration.device_map) \
                    if isinstance(v, torch.Tensor) \
                        else v for k, v in self._inputs.items()}
        else:
            self._inputs = {k: v for k, v in self._inputs.items()}

        return self._inputs

    def load_model(self):
        self._model = \
            VibeVoiceForConditionalGenerationInference.from_pretrained(
                **self.fpm_configuration.to_dict())

    def generate(self, inputs = None):
        if inputs is None:
            inputs = self._inputs

        self._output = self._model.generate(
            **inputs,
            tokenizer=self._processor.tokenizer,
            max_new_tokens=self.vv_configuration.max_new_tokens,
            cfg_scale=self.vv_configuration.cfg_scale
        )

        return self._output

    def process_and_save_output(self, output = None):
        if output is None:
            output = self._output

        generated_speech = output.speech_outputs[0]
        processor_sampling_rate = self._processor.audio_processor.sampling_rate
        save_filename, full_hash = self.vv_configuration.create_save_filename()
        save_path = Path(self.vv_configuration.directory_path_to_save) / \
            save_filename
        self._processor.save_audio(
            generated_speech,
            str(save_path),
            sampling_rate=processor_sampling_rate,
        )
        return save_path, full_hash