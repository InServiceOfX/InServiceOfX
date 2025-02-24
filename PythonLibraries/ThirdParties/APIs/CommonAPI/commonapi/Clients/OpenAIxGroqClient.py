from commonapi.Clients.OpenAIClientWrapper import (
    AsyncOpenAIClientWrapper,
    OpenAIClientWrapper
)

GROQ_BASE_URL = "https://api.groq.com/openai/v1"

class OpenAIxGroqClient(OpenAIClientWrapper):
    def __init__(self, api_key: str):
        """
        https://console.groq.com/docs/openai
        """
        super().__init__(api_key, GROQ_BASE_URL)

class AsyncOpenAIxGroqClient(AsyncOpenAIClientWrapper):
    def __init__(self, api_key: str):
        """
        https://console.groq.com/docs/openai
        """
        super().__init__(api_key, GROQ_BASE_URL)
