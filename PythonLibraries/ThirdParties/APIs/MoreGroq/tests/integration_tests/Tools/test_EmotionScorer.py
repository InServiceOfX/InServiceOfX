from corecode.Utilities import (get_environment_variable, load_environment_file)
from moregroq.Tools.EmotionScorer import (
    score_image_emotion,
    ImageEmotionScore
)
from moregroq.Wrappers import GroqAPIWrapper

load_environment_file()

def test_EmotionScorer_works():
    groq_api_wrapper = GroqAPIWrapper(
        api_key=get_environment_variable("GROQ_API_KEY"))
    
    # Configure the wrapper for emotion scoring
    # E               groq.BadRequestError: Error code: 400 - {'error': {'message': 'The model `llama-3.2-90b-vision-preview` has been decommissioned and is no longer supported. Please refer to https://console.groq.com/docs/deprecations for a recommendation on which model to use instead.', 'type': 'invalid_request_error', 'code': 'model_decommissioned'}}
    #groq_api_wrapper.configuration.model = "llama-3.2-90b-vision-preview"
    # https://console.groq.com/docs/deprecations
    groq_api_wrapper.configuration.model = "meta-llama/llama-4-scout-17b-16e-instruct"
    groq_api_wrapper.configuration.stream = False
    groq_api_wrapper.configuration.temperature = 0.3  # Lower for more consistent scoring
    groq_api_wrapper.configuration.max_completion_tokens = 1024
    
    # Test image and emotion
    image_url = "https://www.michaeldivine.com/wp-content/uploads/2021/01/Station-to-Station-1.jpg"
    emotion = "Horror"
    
    # Get the score using our function that doesn't depend on instructor
    result = score_image_emotion(groq_api_wrapper, image_url, emotion)
    
    # Check that we got a valid result
    assert isinstance(result, ImageEmotionScore)
    assert 0.0 <= result.score <= 1.0
    assert len(result.reasoning) > 0
    
    # Print the results
    print(f"\nEmotion: {emotion}")
    print(f"Score: {result.score}")
    print(f"Reasoning: {result.reasoning}")
