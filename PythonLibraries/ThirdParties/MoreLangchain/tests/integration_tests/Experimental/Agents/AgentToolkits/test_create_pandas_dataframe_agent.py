from datetime import datetime, timedelta

from langchain.agents import (AgentExecutor, AgentType)

from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

from morelangchain.Configuration import GroqGenerationConfiguration
from morelangchain.Groq import create_ChatGroq

from moretransformers.Configurations import (
    Configuration,
    GenerationConfiguration)
from morelangchain.Core.LanguageModels import LocalLlama3

from pathlib import Path

test_data_directory = Path(__file__).resolve().parents[4] / "TestData"

import numpy as np
import pandas as pd

import pytest

# Generate sample data
N = 1000

# Generate dates
start_date = datetime(2022, 1, 1)
dates = [start_date + timedelta(days=i) for i in range(N)]

# Define data categories
makes = [
    'Toyota',
    'Honda',
    'Ford',
    'Chevrolet',
    'Nissan',
    'BMW',
    'Mercedes',
    'Audi',
    'Hyundai',
    'Kia']
models = ['Sedan', 'SUV', 'Truck', 'Hatchback', 'Coupe', 'Van']
colors = ['Red', 'Blue', 'Black', 'White', 'Silver', 'Gray', 'Green']

# Create the dataset
data = {
    'Date': dates,
    'Make': np.random.choice(makes, N),
    'Model': np.random.choice(models, N),
    'Color': np.random.choice(colors, N),
    'Year': np.random.randint(2015, 2023, N),
    'Price': np.random.uniform(20000, 80000, N).round(2),
    'Mileage': np.random.uniform(0, 100000, N).round(0),
    'EngineSize': np.random.choice([1.6, 2.0, 2.5, 3.0, 3.5, 4.0], N),
    'FuelEfficiency': np.random.uniform(20, 40, N).round(1),
    'SalesPerson': np.random.choice(
        ['Alice', 'Bob', 'Charlie', 'David', 'Eva'], N)
}

# Create DataFrame and sort by date
df = pd.DataFrame(data).sort_values('Date')

class SimpleDataAnalysisAgentWithPandas:
    ai_prompt = "\nAI: To answer this question, I need to use Python to analyze the dataframe. I'll use the python_repl_ast tool.\n\nAction: python_repl_ast\nAction Input: "

    def __init__(self, agent, handle_parsing_errors=True):
        """
        See langchain/agents/agent.py for class AgentExecutor(Chain) and
        handle_parsing_errors.
        """
        self.agent = agent
        self.agent.handle_parsing_errors = handle_parsing_errors

    def ask_agent(self, question):
        # .run method is being deprecated in favor of .invoke.
        response = self.agent.invoke({
            "input": question,
            "agent_scratchpad": f"Human: {question}" + self.ai_prompt
        })

        return response

def test_create_pandas_dataframe_agent_value_errors_on_dangerous_code(
        more_transformers_test_data_directory):

    configuration = Configuration(
        more_transformers_test_data_directory / \
            "configuration-llama3.yml")
    generation_configuration = GenerationConfiguration(
        more_transformers_test_data_directory / \
            "generation_configuration-llama3.yml")

    model = LocalLlama3(
        configuration=configuration,
        generation_configuration=generation_configuration)

    with pytest.raises(ValueError):
        agent = create_pandas_dataframe_agent(
            model,
            df,
        verbose=True)        

def test_create_pandas_dataframe_agent_creates_agent_executor(
        more_transformers_test_data_directory):
    """
    See https://github.com/NirDiamant/GenAI_Agents/blob/main/all_agents_tutorials/simple_data_analysis_agent_notebook.ipynb
    """
    configuration = Configuration(
        more_transformers_test_data_directory / \
            "configuration-llama3.yml")
    generation_configuration = GenerationConfiguration(
        more_transformers_test_data_directory / \
            "generation_configuration-llama3.yml")

    model = LocalLlama3(
        configuration=configuration,
        generation_configuration=generation_configuration)

    """
    See langchain-experimental/libs/experimental/langchain_experimental/agents/agent_toolkits/pandas/base.py

    def create_pandas_dataframe_agent(
        llm: LanguageModelLike,
        df: Any,
        agent_type: Union [
            AgentType, Literal[]
        ] = AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose: bool = False,
        ...
        **kwargs: Any,
    ) -> AgentExecutor:
    Args:
        llm: Language model to use for the agent. If agent_type is
            "tool-calling" then llm is expected to support tool calling.
        df: Pandas dataframe or list of Pandas dataframes.
        agent_type: One of "tool-calling" or "openai-tools", "openai-functions",
            or "tool-calling" is recommended over legacy "openai-tools" and
            "openai-functions" types.
    """
    agent = create_pandas_dataframe_agent(
        model,
        df,
        verbose=True,
        allow_dangerous_code=True)

    assert isinstance(agent, AgentExecutor)
    # See
    # langchain/agents/agent.py for class AgentExecutor(Chain):
    agent_input_keys = agent.input_keys
    assert len(agent_input_keys) == 1
    assert agent_input_keys[0] == "input"

question_1 = "What are the column names in this dataset?"
question_2 = "How many rows are in this dataset?"
question_3 = "What is the average prices of cars sold?"

def test_create_pandas_dataframe_agent_invokes_locally_on_question_3(
        more_transformers_test_data_directory):

    configuration = Configuration(
        more_transformers_test_data_directory / \
            "configuration-llama3.yml")
    generation_configuration = GenerationConfiguration(
        more_transformers_test_data_directory / \
            "generation_configuration-llama3.yml")

    model = LocalLlama3(
        configuration=configuration,
        generation_configuration=generation_configuration)

    agent = create_pandas_dataframe_agent(
        model,
        df,
        verbose=True,
        allow_dangerous_code=True)

    agent_with_pandas = SimpleDataAnalysisAgentWithPandas(agent)

    response_3 = agent_with_pandas.ask_agent(question_3)

    assert response_3 is not None

def test_create_pandas_dataframe_agent_invokes_with_groq():

    generation_configuration = GroqGenerationConfiguration(
        test_data_directory / "generation_configuration-groq.yml")

    llm = create_ChatGroq(generation_configuration)

    agent = create_pandas_dataframe_agent(
        llm,
        df,
        verbose=True,
        allow_dangerous_code=True)

    agent_with_pandas = SimpleDataAnalysisAgentWithPandas(agent)

    response_1 = agent_with_pandas.ask_agent(question_1)

    assert response_1 is not None
    if isinstance(response_1, list) or isinstance(response_1, tuple):
        for key in data.keys():
            assert key in response_1
    if isinstance(response_1, str):
        for key in data.keys():
            assert key in response_1

    assert df.shape[0] == 1000
    assert len(df.to_dict().values()) == 10

def test_create_pandas_dataframe_agent_invokes_with_groq_on_questions():

    generation_configuration = GroqGenerationConfiguration(
        test_data_directory / "generation_configuration-groq.yml")

    llm = create_ChatGroq(generation_configuration)

    agent = create_pandas_dataframe_agent(
        llm,
        df,
        verbose=True,
        allow_dangerous_code=True)

    agent_with_pandas = SimpleDataAnalysisAgentWithPandas(agent)

    response_2 = agent_with_pandas.ask_agent(question_2)
    response_3 = agent_with_pandas.ask_agent(question_3)

    assert response_2 is not None
    assert isinstance(response_2, dict)
    assert "1000" in response_2["output"]
    assert response_3 is not None
