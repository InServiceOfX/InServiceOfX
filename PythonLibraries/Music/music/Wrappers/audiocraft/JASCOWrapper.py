from audiocraft.models import JASCO
from audiocraft.data.audio import audio_write

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

