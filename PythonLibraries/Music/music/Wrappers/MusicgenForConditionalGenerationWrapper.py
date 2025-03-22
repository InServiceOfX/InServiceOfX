from transformers import MusicgenForConditionalGeneration

import soundfile as sf

class MusicgenForConditionalGenerationWrapper:
    def __init__(self, configuration):
        self.configuration = configuration
        self.model = MusicgenForConditionalGeneration.from_pretrained(
            configuration.pretrained_model_name_or_path,
            local_files_only=True,
            **configuration.get_model_kwargs())

    def generate(self, inputs, generation_configuration):
        inputs.to(self.configuration.device_map)
        return self.model.generate(
            **inputs,
            **generation_configuration.get_generation_kwargs()).cpu()

    def save_using_soundfile(self, audio_values, output_path, dtype=None):
        audio_values = audio_values.cpu()
        if dtype is not None:
            audio_values = audio_values.to(dtype=dtype)
        sampling_rate = self.model.config.audio_encoder.sampling_rate
        sf.write(
            output_path,
            audio_values.numpy()[0].T,
            sampling_rate)