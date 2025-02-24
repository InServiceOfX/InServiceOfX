"""
https://github.com/groq/groq-api-cookbook/blob/main/tutorials/llama3-stock-market-function-calling/llama3-stock-market-function-calling.ipynb
"""

from corecode.Utilities import (get_environment_variable, load_environment_file)
from commonapi.Messages import (
    create_system_message,
    create_user_message
)

from moregroq.Wrappers.ChatCompletionConfiguration import (
    FunctionDefinition,
    FunctionParameters,
    ParameterProperty,
    Tool)

from moregroq.Wrappers import GroqAPIWrapper

load_environment_file()
