from corecode.Utilities import get_environment_variable

from groq import (AsyncGroq, Groq)

class GroqAPIInterface:
    def __init__(self):
        self.client = Groq(api_key=get_environment_variable("GROQ_API_KEY"))

    def chat_completion(self, messages, model, **kwargs):
        return self.client.chat.completions.create(
            messages=messages,
            model=model,
            **kwargs)
