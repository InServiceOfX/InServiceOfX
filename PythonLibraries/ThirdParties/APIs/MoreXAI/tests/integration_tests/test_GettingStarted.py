from xai_sdk import Client
from xai_sdk.chat import user, image

from corecode.Utilities import (get_environment_variable, load_environment_file)
load_environment_file()

def test_Grok_analyzes_image():

    client = Client(
        api_key=get_environment_variable("XAI_API_KEY"),
        # Override default timeout with longer timeout for reasoning models.
        timeout=3600,
    )

    chat = client.chat.create(model="grok-4")
    chat.append(
        user(
            "What's in this image?",
            image("https://science.nasa.gov/wp-content/uploads/2023/09/web-first-images-release.png")
        )
    )

    response = chat.sample()
    print(type(response))
    print(response.content)
    assert(isinstance(response.content, str))