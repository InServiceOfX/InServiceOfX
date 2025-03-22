from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

class MusicGenWrapper:
    def __init__(self, configuration, generation_configuration):

        self.configuration = configuration
        self.generation_configuration = generation_configuration

        self.model = MusicGen.get_pretrained(
            configuration.pretrained_model_name_or_path,
            device=configuration.device_map)

        self.model.set_generation_params(
            **generation_configuration.get_generation_kwargs())

    def set_generation_parameters(self, generation_configuration):
        self.generation_configuration = generation_configuration
        self.model.set_generation_params(
            **generation_configuration.get_generation_kwargs())

import torchaudio

class MusicGenWrapperForMelodyAndStyle(MusicGenWrapper):
    def __init__(self, configuration, generation_configuration):
        super().__init__(configuration, generation_configuration)

    def generate_with_chroma(self, descriptions, audio_file_path):
        """
        Returns tp.Union[torch.Tensor, tp.Tuple[torch.Tensor, torch.Tensor]]
        """
        melody, sample_rate = torchaudio.load(audio_file_path)

        return self.model.generate_with_chroma(
            descriptions,
            melody.to(self.configuration.device_map)[None].expand(
                len(descriptions),
                -1,
                -1),
            sample_rate)

    def save_wav(self, wav, filename_prefix):
        for idx, one_wave in enumerate(wav):
            audio_write(
                filename_prefix + f'{idx}.wav',
                one_wave.cpu(),
                self.model.sample_rate,
                strategy="loudness")