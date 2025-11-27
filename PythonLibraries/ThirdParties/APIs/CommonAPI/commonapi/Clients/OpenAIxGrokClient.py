from commonapi.Clients.OpenAIClientWrapper import (
    AsyncOpenAIClientWrapper,
    OpenAIClientWrapper
)

# Guessed at this URL from the documentation:
# https://docs.x.ai/docs/tutorial
# and where it says
# curl https://api.x.ai/v1/chat/completions \
# and guess that we remove chat/completions from the URL.
GROK_BASE_URL = "https://api.x.ai/v1"

class OpenAIxGrokClient(OpenAIClientWrapper):
    def __init__(self, api_key: str):
        """
        https://docs.x.ai/docs/tutorial
        """
        super().__init__(api_key, GROK_BASE_URL)

class AsyncOpenAIxGrokClient(AsyncOpenAIClientWrapper):
    def __init__(self, api_key: str):
        """
        https://docs.x.ai/docs/tutorial
        """
        super().__init__(api_key, GROK_BASE_URL)
