from corecode.Utilities import (load_environment_file)
from pydantic_ai import Agent

load_environment_file()

def test_create_Agents_on_groq_models():
    # https://console.groq.com/docs/models
    agent = Agent('groq:gemma2-9b-it')
    assert agent is not None
    agent = Agent('groq:meta-llama/llama-guard-4-12b')
    assert agent is not None

    agent = Agent('groq:llama-3.3-70b-versatile')
    assert agent is not None
    agent = Agent('groq:llama-3.1-8b-instant')
    assert agent is not None
    agent = Agent('groq:llama3-70b-8192')
    assert agent is not None
    agent = Agent('groq:llama3-8b-8192')
    assert agent is not None
    agent = Agent('groq:whisper-large-v3')
    assert agent is not None
    agent = Agent('groq:whisper-large-v3-turbo')
    assert agent is not None
    # TODO: Got this error:
    # pydantic_ai.exceptions.UserError: Unknown model: distil-whisper-large-v3-en
    #agent = Agent('distil-whisper-large-v3-en')
    #assert agent is not None

def test_create_Agents_on_Anthropic():
    # https://docs.anthropic.com/en/docs/about-claude/models/overview
    agent = Agent('anthropic:claude-opus-4-20250514')
    assert agent is not None
    agent = Agent('anthropic:claude-sonnet-4-20250514')
    assert agent is not None
    agent = Agent('anthropic:claude-3-7-sonnet-20250219')
    assert agent is not None
    agent = Agent('anthropic:claude-3-7-sonnet-latest')
    assert agent is not None
    agent = Agent('anthropic:claude-3-5-haiku-20241022')
    assert agent is not None
    agent = Agent('anthropic:claude-3-5-haiku-latest')
    assert agent is not None
    agent = Agent('anthropic:claude-3-5-sonnet-20241022')
    assert agent is not None
    agent = Agent('anthropic:claude-3-5-sonnet-latest')
    assert agent is not None
    
