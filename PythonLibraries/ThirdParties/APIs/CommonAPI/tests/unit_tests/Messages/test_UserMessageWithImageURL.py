from commonapi.Messages import UserMessageWithImageURL

def test_UserMessageWithImageURL_init_works():
    user_message_with_image_url = UserMessageWithImageURL(content=[])
    assert user_message_with_image_url.content == []
    assert user_message_with_image_url.role == "user"

def test_UserMessageWithImageURL_from_text_and_image_works():
    user_message_with_image_url = UserMessageWithImageURL.from_text_and_image(
        "What is the meaning of life, the universe, and everything?",
        "https://example.com/image.png")
    assert user_message_with_image_url.content == [
        {
            "type": "text",
            "text": "What is the meaning of life, the universe, and everything?"
        },
        {
            "type": "image_url",
            "image_url": {
                "url": "https://example.com/image.png"
            }
        }
    ]
    assert user_message_with_image_url.role == "user"

def test_UserMessageWithImageURL_to_dict_works():
    question = "What's in this image?"
    image_url = \
        "https://science.nasa.gov/wp-content/uploads/2023/09/web-first-images-release.png"
    user_message_with_image_url = UserMessageWithImageURL.from_text_and_image(
        question, image_url)
    assert user_message_with_image_url.to_dict() == {
        "role": "user",
        "content": [
            {"type": "text", "text": question},
            {"type": "image_url", "image_url": {"url": image_url}}
        ]
    }