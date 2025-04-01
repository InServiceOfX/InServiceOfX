from moregroq.Messages import create_vision_message
from pydantic import BaseModel, Field

class ImageEmotionScore(BaseModel):
    score: float = Field(
        description=\
            "A score from 0.0 to 1.0 indicating how relevant the emotion is to the image",
        ge=0.0,
        le=1.0
    )
    reasoning: str = Field(
        description="Brief explanation of why this score was assigned"
    )

# Combine the system and user prompts
IMAGE_EMOTION_SCORE_PROMPT = (
    "You are an art emotion analyzer that returns precise numerical scores. "
    "You are an expert at judging and scoring the emotions evoked in a piece "
    "of art, from 0.0 to 1.0, where 1.0 means the emotion is strongly present "
    "and 0.0 means it's completely absent. "
    "Provide a detailed analysis and score the relevancy of this image to the emotion: {emotion}. "
    "Your response must be a valid JSON object with exactly two fields: "
    "'score' (a float between 0.0 and 1.0) and 'reasoning' (a string explaining your score)."
)

def create_emotion_scoring_message(image_url: str, emotion: str):
    """
    Create a vision message for scoring a specific emotion in an image.
    
    Args:
        image_url: URL of the image to analyze
        emotion: The emotion to score (e.g., "happiness", "melancholy")
        
    Returns:
        A list containing the vision message
    """
    # Substitute the emotion into the template
    prompt = IMAGE_EMOTION_SCORE_PROMPT.format(emotion=emotion)
    
    # Create the vision message with the formatted prompt
    vision_message = create_vision_message(
        text=prompt,
        image_urls=image_url
    )
    
    # Return a list with just the vision message
    return [vision_message]

def score_image_emotion(groq_api_wrapper, image_url: str, emotion: str):
    """
    Args:
        groq_api_wrapper: Configured GroqAPIWrapper instance
        image_url: URL of the image to analyze
        emotion: The emotion to score (e.g., "happiness", "melancholy")
        
    Returns:
        ImageEmotionScore with score and reasoning
    """
    # Save existing configuration
    original_response_format = groq_api_wrapper.configuration.response_format
    
    # Configure for JSON output
    groq_api_wrapper.configuration.response_format = {"type": "json_object"}
    
    # Create the message(s)
    messages = create_emotion_scoring_message(image_url, emotion)
    
    # Get the response
    result = groq_api_wrapper.create_chat_completion(messages)
    
    # Parse the JSON content
    import json
    json_str = result.choices[0].message.content
    data = json.loads(json_str)
    
    # Restore original configuration
    groq_api_wrapper.configuration.response_format = original_response_format
    
    return ImageEmotionScore(score=data["score"], reasoning=data["reasoning"])
