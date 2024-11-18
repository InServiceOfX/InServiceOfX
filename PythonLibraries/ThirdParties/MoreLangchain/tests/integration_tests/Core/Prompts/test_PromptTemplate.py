from moretransformers.Configurations import (
    Configuration,
    GenerationConfiguration)

from morelangchain.Core.LanguageModels import LocalLlama3

from langchain_core.prompts import PromptTemplate

class QuestionAnsweringAgent:
    def __init__(self, chain):
        self.chain = chain
    def get_answer(self, question):
        """
        Get an answer to the given question using the QA chain.
        """
        input_variables = {"question": question}
        response = self.chain.invoke(input_variables)
        return response

def test_PromptTemplate(more_transformers_test_data_directory):
    """
    See libs/core/langchain_core/prompts/prompt.py for implementation of
    class PromptTemplate(StringPromptTemplate)
    Code comments says:
    Prompt template for a language model.

    A prompt template consists of a string template. It accepts a set of
    parameters from user that can be used to generate a prompt for a language
    model.

    https://github.com/NirDiamant/GenAI_Agents/blob/main/all_agents_tutorials/simple_question_answering_agent.ipynb
    for the example.
    """
    template = """
    You are a helpful AI assistant. Your task is to answer the user's question
    to the best of your ability.

    User's question: {question}

    Please provide a clear and concise answer:
    """
    configuration = Configuration(
        more_transformers_test_data_directory / \
            "configuration-llama3.yml")
    generation_configuration = GenerationConfiguration(
        more_transformers_test_data_directory / \
            "generation_configuration-llama3.yml")

    agent = LocalLlama3(
        configuration=configuration,
        generation_configuration=generation_configuration)

    prompt_template = PromptTemplate(
        template=template,
        input_variables=["question"])

    qa_chain = prompt_template | agent

    question_answering_agent = QuestionAnsweringAgent(qa_chain)

    question = "What is the capital of France?"
    answer = question_answering_agent.get_answer(question)
    assert "Paris" in answer


