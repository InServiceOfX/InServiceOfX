from corecode.Utilities import load_environment_file
from datetime import date
from pydantic_ai import Agent, RunContext

load_environment_file()

def test_Agent_system_prompt_decorator():
    # https://ai.pydantic.dev/agents/?utm_source=chatgpt.com#system-prompts
    agent = Agent(
        'groq:gemma2-9b-it',
        system_prompt="You are a helpful assistant."
    )

    @agent.system_prompt
    def add_the_date() -> str:  
        return f'The date is {date.today()}.'

    result = agent.run_sync('What is the date?')
    assert result is not None
    assert isinstance(result.output, str)
    assert result.output.startswith("The date is")

def test_Agent_system_prompt_decorator_and_RunContext():
    # https://ai.pydantic.dev/agents/?utm_source=chatgpt.com#system-prompts
    agent = Agent(
        'groq:gemma2-9b-it',
        system_prompt="Use the customer's name while replying to them."
    )
    
    @agent.system_prompt  
    def add_the_users_name(ctx: RunContext[str]) -> str:
        return f"The user's name is {ctx.deps}."

    @agent.system_prompt
    def add_the_date() -> str:  
        return f'The date is {date.today()}.'

    result = agent.run_sync('What is the date?', deps='Frank')
    assert result is not None
    assert result.output is not None
    assert "date is" in result.output.lower() or \
        "today is" in result.output.lower()
    assert "frank" in result.output.lower()

def test_Agent_system_prompt():

    system_prompt = "You are a helpful assistant."
    agent = Agent(
        'groq:gemma2-9b-it',
        system_prompt=system_prompt
    )
    result = agent.run_sync('What is the capital of France?')
    assert result is not None
    assert result.output is not None
    assert "paris" in result.output.lower()
    assert set(agent._system_prompts) == set([system_prompt])

    system_prompt_2 = \
        "You are Samuel L. Jackson and you speak like Samuel L. Jackson and is grumpy."
    # Demonstrate that this does nothing and it's not clear how to directly
    # change the system prompt other than by the decorator.
    agent.system_prompt = system_prompt_2
    result = agent.run_sync(
        'What do they call a quarter pounder with cheese in France?')
    assert result is not None
    assert result.output is not None
    assert "royal" in result.output.lower()
    assert set(agent._system_prompts) == set([system_prompt])

    #print(result.output)