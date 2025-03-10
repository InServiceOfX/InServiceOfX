from textwrap import dedent
from typing import Final

class PromptData:
    # https://github.com/groq/groq-api-cookbook/blob/main/tutorials/mixture-of-agents/mixture_of_agents.ipynb

    HELPFUL_SYSTEM_TEMPLATE: Final[str] = \
        "You are a helpful assistant.\n{helper_response}"

    REFERENCE_SYSTEM_PROMPT_TEMPLATE: Final[str] = dedent("""\
    You have been provided with a set of responses from various open-source models to the latest user query. 
    Your task is to synthesize these responses into a single, high-quality response. 
    It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. 
    Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. 
    Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.
    Responses from models:
    {responses}
    """)

    EXPERT_PLANNER_TEMPLATE: Final[str] = (
        "You are an expert planner agent. Break down and plan out how you can "
        "answer the user's question {helper_response}")

    THOUGHT_AND_QUESTION_RESPONSE_TEMPLATE: Final[str] = (
        "Respond with a thought and then your response to the question. "
        "{helper_response}")

    CHAIN_OF_THOUGHT_TEMPLATE: Final[str] = \
        "Think through your response step by step. {helper_response}"    

    HELPFUL_BOB_TEMPLATE: Final[str] = \
        "You are a helpful assistant named Bob.\n{helper_response}"