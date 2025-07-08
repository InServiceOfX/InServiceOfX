
from commonapi.Messages import (
    AssistantMessage
    ConversationAndSystemMessages,
    UserMessage)

from corecode.FileIO import JSONFile

def test_ConversationAndSystemMessages_init():
    # https://prompts.chat/
    system_prompt = (
        "I want you to act as a prompt generator for Midjourney’s artificial "
        "intelligence program. Your job is to provide detailed and creative "
        "descriptions that will inspire unique and interesting images from the "
        "AI. Keep in mind that the AI is capable of understanding a wide range of language and can interpret abstract concepts, so feel free to be as imaginative and descriptive as possible. For example, you could describe a scene from a futuristic city, or a surreal landscape filled with strange creatures. The more detailed and imaginative your description, the more interesting the resulting image will be. Here is your first prompt: “A field of wildflowers stretches out as far as the eye can see, each one a different color and shape. In the distance, a massive tree towers over the landscape, its branches reaching up to the sky like tentacles.”"
    )

    conversation_and_system_messages = ConversationAndSystemMessages()
    conversation_and_system_messages.add_default_system_message()
    conversation_and_system_messages.add_system_message(system_prompt)


