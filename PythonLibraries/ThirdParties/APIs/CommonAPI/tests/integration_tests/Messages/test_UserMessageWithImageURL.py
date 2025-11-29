from corecode.Utilities import (get_environment_variable, load_environment_file)
load_environment_file()

from commonapi.Clients.OpenAIxGrokClient import OpenAIxGrokClient
from commonapi.Clients.OpenAIxGroqClient import OpenAIxGroqClient
from commonapi.Messages import UserMessageWithImageURL

question = "What's in this image?"
image_url = \
    "https://science.nasa.gov/wp-content/uploads/2023/09/web-first-images-release.png"

def test_UserMessageWithImageURL_with_groq():
    client = OpenAIxGroqClient(get_environment_variable("GROQ_API_KEY"))
    
    client.clear_chat_completion_configuration()
    client.configuration.model = "meta-llama/llama-4-scout-17b-16e-instruct"

    messages = [
        UserMessageWithImageURL.from_text_and_image(
            question,
            image_url).to_dict()
    ]

    response = client.create_chat_completion(messages)
    # Example (actual) response:
    # The image depicts a breathtaking scene of a star-forming region in space, showcasing a vast array of colors and celestial objects. The main subject appears to be a nebula, characterized by its vibrant orange and yellow hues, with wispy tendrils of gas and dust.

    # **Key Features:**

    # * **Nebula:** A large, sprawling nebula dominates the center of the image, featuring a complex network of gas and dust.
    # * **Stars:** Numerous stars are scattered throughout the image, ranging from small, faint points to larger, brighter ones that cast a warm glow.
    # * **Color Scheme:** The dominant colors are shades of orange, yellow, and blue, which are likely indicative of different elements or temperatures within the nebula.
    # * **Background:** The background is a deep blue, representing the vastness of space.

    # **Overall Impression:**

    # The image presents a stunning visual representation of a star-forming region, highlighting the intricate beauty and complexity of celestial bodies in our universe. The combination of vibrant colors and diverse celestial objects creates a captivating scene that invites exploration and contemplation.
    print(response.choices[0].message.content)
    # <class 'openai.types.chat.chat_completion.ChatCompletion'>
    print(type(response))
    # 1
    print(len(response.choices))
    # <class 'openai.types.chat.chat_completion_message.ChatCompletionMessage'>
    print(type(response.choices[0].message))
    print(response.usage)

    model = "meta-llama/llama-4-maverick-17b-128e-instruct"
    client.clear_chat_completion_configuration()
    client.configuration.model = model

    messages = [
        UserMessageWithImageURL.from_text_and_image(
            question,
            image_url).to_dict()
    ]

    response = client.create_chat_completion(messages)
    # Example (actual) response:
    # The image depicts a vibrant and intricate nebula, characterized by a predominantly orange hue with subtle blue undertones. The nebula's texture is reminiscent of rugged mountains or waves, with a sense of depth and dimensionality. The surrounding space is filled with numerous stars, adding to the overall sense of celestial wonder.

    # **Key Features:**

    # * **Nebula:** The central focus of the image, the nebula is a vast, interstellar cloud of gas and dust.
    # * **Color:** The nebula's dominant color is orange, with hints of blue visible in certain areas.
    # * **Texture:** The nebula's surface appears rugged and mountainous, with undulating peaks and valleys.
    # * **Stars:** The surrounding space is densely populated with stars, ranging in size and brightness.
    # * **Background:** The background of the image is a deep shade of blue, gradating to black towards the edges.

    # **Overall Impression:**

    # The image presents a breathtaking view of a nebula, showcasing its intricate structure and the surrounding celestial environment. The combination of vibrant colors, textured surfaces, and starry skies creates a visually stunning representation of the cosmos.
    print(response.choices[0].message.content)
    print(response.usage)


def test_UserMessageWithImageURL_with_grok():
    client = OpenAIxGrokClient(get_environment_variable("XAI_API_KEY"))
    
    client.clear_chat_completion_configuration()
    client.configuration.model = "grok-4"

    messages = [
        UserMessageWithImageURL.from_text_and_image(
            question,
            image_url).to_dict()
    ]

    response = client.create_chat_completion(messages)

    # Example (actual) response:
    # # Based on the image you provided, here's a description and analysis of what's in it:

    # ### Visual Description
    # - **Overall Scene**: This appears to be a stunning astronomical photograph (or composite image) of a cosmic nebula—a vast cloud of interstellar gas and dust in space. The image shows a dramatic, wavy ridge or "mountain" of glowing, brownish-orange clouds dominating the foreground, set against a deep blue-black starry sky. The clouds have a textured, almost ethereal quality, with hints of blue and purple hues mixing in, suggesting ionized gases illuminated by nearby stars.
    # - **Key Elements**:
    #   - **Cloud Formation**: The central feature is a large, undulating structure resembling a cosmic cliff or wave, with peaks and valleys. It's richly colored in warm tones (oranges, browns, and golds), likely representing dense regions of molecular hydrogen and dust where new stars might be forming.
    #   - **Stars**: Numerous bright stars are scattered throughout, twinkling against the dark background. Some are embedded within or shining through the nebula, creating a sense of depth and vastness. There are also faint star clusters or points of light in the distance.
    #   - **Background**: A starry field with a gradient from deep blue (possibly representing ionized regions) to black space, dotted with distant stars and subtle nebular glows.
    # - **Style and Quality**: This looks like a high-resolution, color-enhanced image typical of those captured by space telescopes like the Hubble Space Telescope (HST) or James Webb Space Telescope (JWST). The colors are not "natural" but processed to highlight different wavelengths of light (e.g., infrared or visible spectrum) for scientific and aesthetic purposes.

    # ### Likely Identification
    # This image strongly resembles a view of the **Eagle Nebula (Messier 16 or M16)**, specifically a region often called the "Pillars of Creation" or a similar stellar nursery. The Eagle Nebula is located about 7,000 light-years away in the constellation Serpens and is famous for its towering pillars of gas and dust, sculpted by the radiation from young, hot stars. However:

    # - The coloring and angle here seem like a variant or artistic rendering—perhaps from Hubble's archives or a processed version emphasizing certain features. If it's not the Eagle Nebula, it could be something similar like the Carina Nebula or Orion Nebula, but the ridge-like structure matches Eagle closely.
    # - If this is from a specific source (e.g., NASA/ESA Hubble imagery), it's one of the most iconic space photos ever taken, first popularized in 1995 and revisited in higher detail in later years.

    # If this image is from a particular website, article, or context (e.g., a screenshot or meme), feel free to provide more details for a more precise identification! If you're asking about something specific like hidden objects or Easter eggs in the image, let me know—I don't see anything unusual beyond the cosmic elements. 😊
    print(response.choices[0].message.content)
    print(response.usage)
