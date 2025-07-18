from pydantic_ai.settings import ModelSettings

class ToAndFromModelSettings:

    @staticmethod
    def from_openai_chat_completion_configuration(configuration):
        keyword_arguments = {}
        if configuration.max_tokens is not None:
            keyword_arguments["max_tokens"] = configuration.max_tokens
        if configuration.temperature is not None:
            keyword_arguments["temperature"] = configuration.temperature
        if configuration.parallel_tool_calls is not None:
            keyword_arguments["parallel_tool_calls"] = \
                configuration.parallel_tool_calls
        if configuration.stop is not None:
            keyword_arguments["stop_sequences"] = configuration.stop
        return ModelSettings(**keyword_arguments)

    @staticmethod
    def from_groq_chat_completion_configuration(configuration):
        keyword_arguments = {}
        if configuration.max_tokens is not None:
            keyword_arguments["max_tokens"] = configuration.max_tokens
        if configuration.temperature is not None:
            keyword_arguments["temperature"] = configuration.temperature
        if configuration.parallel_tool_calls is not None:
            keyword_arguments["parallel_tool_calls"] = \
                configuration.parallel_tool_calls
        if configuration.stop is not None:
            keyword_arguments["stop_sequences"] = configuration.stop
        return ModelSettings(**keyword_arguments)