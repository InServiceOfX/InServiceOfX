from audiocraft.models import JASCO
from audiocraft.data.audio import audio_write
from music.Wrappers.audiocraft.JASCOGenerateMusicInputs \
    import JASCOGenerateMusicInputs

class JASCOWrapper:
    def __init__(self, configuration, generation_configuration):

        self.configuration = configuration
        self.generation_configuration = generation_configuration

        self.model = JASCO.get_pretrained(
            configuration.pretrained_model_name_or_path,
            chords_mapping_path=configuration.chords_mapping_path,
            device=configuration.device_map)

        self.model.set_generation_params(
            **generation_configuration.get_generation_kwargs())

    def set_generation_parameters(self, generation_configuration):
        self.generation_configuration = generation_configuration
        self.model.set_generation_params(
            **generation_configuration.get_generation_kwargs())

    def generate_music(self, generation_inputs: JASCOGenerateMusicInputs):
        return self.model.generate_music(**generation_inputs.to_dict())
        
    def save_audio_via_audiocraft(
        self,
        audio_tensor,
        filename_prefix,
        sample_rate=None):
        """Save audio tensor to file"""
        if sample_rate is None:
            sample_rate = self.model.sample_rate
            
        # Handle both batched and single outputs
        if audio_tensor.dim() > 2:
            # Batched output
            for i, sample in enumerate(audio_tensor):
                audio_write(
                    f"{filename_prefix}_{i}",
                    sample.cpu(),
                    sample_rate,
                    strategy="loudness",
                    loudness_compressor=True
                )
        else:
            # Single output
            audio_write(
                filename_prefix,
                audio_tensor.cpu(),
                sample_rate,
                strategy="loudness",
                loudness_compressor=True
            )

