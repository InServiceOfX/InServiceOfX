import pytest
from groq import (AsyncGroq, Groq)

from corecode.Utilities import load_environment_file
from corecode.Utilities.load_environment_file import get_environment_variable

from pathlib import Path

load_environment_file()

test_image_data_directory = Path(__file__).parents[4] / "ThirdParties" / \
    "MoreInsightFace" / "tests" / "TestData" / "Images"

# https://console.groq.com/docs/quickstart
def test_groq():
    client = Groq(api_key=get_environment_variable("GROQ_API_KEY"))
    messages = [
        {
            "role": "user",
            "content": "Explain the importance of fast language models",
        }
    ]
    model = "llama3-8b-8192"

    chat_completion = client.chat.completions.create(
        messages=messages,
        model=model
    )

    assert len(chat_completion.choices) == 1
    assert isinstance(chat_completion.choices[0].message.content, str)
    # Uncomment to print the response
    #print(chat_completion.choices[0].message.content)

@pytest.mark.asyncio
async def test_AsyncGroq():
    client = AsyncGroq(api_key=get_environment_variable("GROQ_API_KEY"))

    messages = [
        {
            "role": "user",
            "content": "Explain the importance of low latency LLMs",
        }
    ]
    model = "llama3-8b-8192"

    # In groq-python/src/groq/_client.py, class AsyncGroq(AsyncAPIClient)
    # defines self.chat = resources.AsyncChat(self). AsynChat is defined in
    # src/groq/resources/chat/chat.py where completions is a @cached_property
    # and AsyncCompletions is defined in
    # src/groq/resources/chat/completions.py which defines
    # @overload
    # async def create(
    #   self,
    #   *,
    #   messages: List[ChatCompletionMessageParam],
    #   model: str,
    #   max_tokens: Optional[int] | NotGiven = NOT_GIVEN
    #   stop: Union[Original[str], List[str], None] | NotGiven = NOT_GIVEN
    #   stream: Optional[Literal[False]] | NotGiven = NOT_GIVEN
    #   temperature: Optional[float] | NotGiven = NOT_GIVEN
    #   top_p: Optional[float] | NotGiven = NOT_GIVEN
    #   user: Optional[str] | NotGiven = NOT_GIVEN
    # ) -> ChatCompletion:
    #
    # max_tokens: Max number of tokens that can be generated in the chat
    # completion. Total length of input tokens and generated tokens limited by
    # model's context length.
    #
    # stop: Up to 4 sequences where the API will stop generating further tokens.
    # The returned text will not contain the stop sequence.
    #   
    # stream: If set, partial messages deltas will be sent. Tokens will be sent
    # as data-only as they become available, with stream terminated by a
    # `data: [DONE]` message.
    #
    # temperature: What sampling temperature to use, between 0 and 2. Higher
    # values like 0.8 will make output more random, while lower values like 0.2
    # will make it more focused and deterministic.  We generally recommend altering
    # this or top_p but not both.
    #
    # top_p: An alternative to sampling with temperature, called nucleus
    # sampling, where model considers results of tokens with top_p probabilty
    # mass. So 0.1 means only tokens comprising the top 10% probability mass are
    # considered. We generally recommend altering this or temperature but not
    # both.
    chat_completion = await client.chat.completions.create(
        messages=messages,
        model=model,
        temperature=1.0,
        max_tokens=1024,
        top_p=1.0,
        stream=False,
        stop=None,
    )

    assert len(chat_completion.choices) == 1
    assert isinstance(chat_completion.choices[0].message.content, str)
    # Uncomment to print the response
    #print(chat_completion.choices[0].message.content)
