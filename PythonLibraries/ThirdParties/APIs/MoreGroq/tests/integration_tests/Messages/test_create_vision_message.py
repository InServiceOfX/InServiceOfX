from corecode.Utilities import (get_environment_variable, load_environment_file)

from moregroq.Messages import create_vision_message
from moregroq.Wrappers import GroqAPIWrapper

load_environment_file()

def test_create_vision_message_works():
    groq_api_wrapper = GroqAPIWrapper(
        api_key=get_environment_variable("GROQ_API_KEY"))

    message = create_vision_message(
        text="What emotions is this image conveying?",
        image_urls="https://www.michaeldivine.com/wp-content/uploads/2021/01/Station-to-Station-1.jpg",)

    groq_api_wrapper.configuration.model = "llama-3.2-90b-vision-preview"
    groq_api_wrapper.configuration.stream = False
    groq_api_wrapper.configuration.temperature = 1
    groq_api_wrapper.configuration.max_completion_tokens = 1024
    groq_api_wrapper.configuration.top_p = 1
    groq_api_wrapper.configuration.stop = None

    messages = [message,] 

    emotion_response = groq_api_wrapper.create_chat_completion(messages)

    print(emotion_response.choices[0].message.content)
    print(emotion_response)