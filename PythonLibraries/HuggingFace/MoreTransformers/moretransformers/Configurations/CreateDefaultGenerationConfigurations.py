from .GenerationConfiguration import GenerationConfiguration

class CreateDefaultGenerationConfigurations:

    @staticmethod
    def for_HuggingFaceTB_SmolLM3_3B() -> GenerationConfiguration:
        generation_configuration = GenerationConfiguration()
        generation_configuration.do_sample = True
        # temperature, top_p suggested by
        # https://huggingface.co/HuggingFaceTB/SmolLM3-3B
        generation_configuration.temperature = 0.6
        generation_configuration.top_p = 0.95
        # max_new_tokens is set to 65536, which is the default value for
        # max_new_tokens in the GenerationConfiguration class.
        generation_configuration.max_new_tokens = 65536
        return generation_configuration

    def for_Menlo_Lucy_128k() -> GenerationConfiguration:
        generation_configuration = GenerationConfiguration()
        generation_configuration.do_sample = True
        # temperature, top_p, top_k, min_p suggested by
        # https://huggingface.co/Menlo/Lucy-128k
        generation_configuration.temperature = 0.7
        generation_configuration.top_p = 0.9
        generation_configuration.top_k = 20
        generation_configuration.min_p = 0.0
        # max_position_embeddings found to be 131072.
        generation_configuration.max_new_tokens = 131072
        return generation_configuration