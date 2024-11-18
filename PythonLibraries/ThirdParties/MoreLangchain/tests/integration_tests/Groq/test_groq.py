"""
https://python.langchain.com/docs/integrations/chat/groq/
"""
from corecode.Utilities import load_environment_file

from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate

from morelangchain.Configuration import GroqGenerationConfiguration

from pathlib import Path

test_data_directory = Path(__file__).resolve().parents[2] / "TestData"

load_environment_file()

def test_ChatGroq_inits():

    generation_configuration = GroqGenerationConfiguration(
        test_data_directory / "generation_configuration-groq.yml")

    assert generation_configuration.timeout == None
    assert generation_configuration.max_new_tokens == None

    # See https://console.groq.com/docs/models for supported models.
    llm = ChatGroq(
        model="llama-3.1-70b-versatile",
        temperature=generation_configuration.temperature,
        max_tokens=generation_configuration.max_new_tokens,
        timeout=generation_configuration.timeout,
        max_retries=generation_configuration.max_retries
    )

    assert isinstance(llm, ChatGroq)

def test_ChatGroq_invokes():

    generation_configuration = GroqGenerationConfiguration(
        test_data_directory / "generation_configuration-groq.yml")

    # See https://console.groq.com/docs/models for supported models.
    llm = ChatGroq(
        model="llama-3.1-70b-versatile",
        temperature=generation_configuration.temperature,
        max_tokens=generation_configuration.max_new_tokens,
        timeout=generation_configuration.timeout,
        max_retries=generation_configuration.max_retries
    )

    messages = [
        (
            "system",
            "You are a helpful assistant that translates English to French. Translate the user sentence.",
        ),
        ("human", "I love programming.")
    ]

    ai_msg = llm.invoke(messages)

    assert isinstance(ai_msg, AIMessage)
    assert "J'aime" in ai_msg.content or "J'adore" in ai_msg.content

def test_ChatGroq_chains():

    generation_configuration = GroqGenerationConfiguration(
        test_data_directory / "generation_configuration-groq.yml")

    # See https://console.groq.com/docs/models for supported models.
    llm = ChatGroq(
        model="llama-3.1-70b-versatile",
        temperature=generation_configuration.temperature,
        max_tokens=generation_configuration.max_new_tokens,
        timeout=generation_configuration.timeout,
        max_retries=generation_configuration.max_retries
    )


    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that translates {input_language} to {output_language}."),
        ("human", "{input}"),
    ])

    chain = prompt | llm

    result = chain.invoke(
        {
            "input_language": "English",
            "output_language": "German",
            "input": "I love programming."
        }
    )

    assert isinstance(result, AIMessage)
    assert "Ich liebe" in result.content
    assert "Programmieren" in result.content