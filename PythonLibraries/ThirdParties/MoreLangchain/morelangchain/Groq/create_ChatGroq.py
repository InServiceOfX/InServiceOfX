from corecode.Utilities import load_environment_file
from langchain_groq import ChatGroq
from morelangchain.Configuration import GroqGenerationConfiguration

def create_ChatGroq(generation_configuration: GroqGenerationConfiguration):
    load_environment_file()
    return ChatGroq(
        model="llama-3.1-70b-versatile",
        temperature=generation_configuration.temperature,
        max_tokens=generation_configuration.max_new_tokens,
        timeout=generation_configuration.timeout,
        max_retries=generation_configuration.max_retries
    )
