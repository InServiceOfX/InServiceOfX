from commonapi.Utilities.StringManipulation import FormatString
from textwrap import dedent

def test_FormatString_for_no_values():
    """See
    https://github.com/groq/groq-api-cookbook/blob/main/tutorials/mixture-of-agents/mixture_of_agents.ipynb"""
    system_prompt = "You are a helpful assistant.\n{helper_response}"
    format_string = FormatString(system_prompt)

    assert format_string.placeholders == ["helper_response"]
    assert format_string.get_formatted_string() == system_prompt

    REFERENCE_SYSTEM_PROMPT = dedent("""\
    You have been provided with a set of responses from various open-source models to the latest user query. 
    Your task is to synthesize these responses into a single, high-quality response. 
    It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. 
    Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. 
    Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.
    Responses from models:
    {responses}
    """)
    format_string = FormatString(REFERENCE_SYSTEM_PROMPT)

    assert format_string.placeholders == ["responses"]
    assert format_string.get_formatted_string() == REFERENCE_SYSTEM_PROMPT

