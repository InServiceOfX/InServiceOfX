from morelangchain.Core.ChatHistoryWrappers import OldStyleChatHistoryStore
from morelangchain.Core.LanguageModels import LocalLlama3

from moretransformers.Configurations import (
    Configuration,
    GenerationConfiguration)

from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder)

from langchain_core.runnables.history import RunnableWithMessageHistory

def test_RunnableWithMessageHistory_with_LocalLlama3(
        more_transformers_test_data_directory):
    configuration = Configuration(
        more_transformers_test_data_directory / \
            "configuration-llama3.yml")
    generation_configuration = GenerationConfiguration(
        more_transformers_test_data_directory / \
            "generation_configuration-llama3.yml")

    agent = LocalLlama3(
        configuration=configuration,
        generation_configuration=generation_configuration)
        
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])

    chain = prompt | agent

    history_store = OldStyleChatHistoryStore()

    chain_with_history = RunnableWithMessageHistory(
        chain,
        history_store.get_chat_history,
        input_messages_key="input",
        history_messages_key="history")

    session_id = "user_123"

    question_1 = "Hello! How are you?"

    response1 = chain_with_history.invoke(
        {"input": question_1},
        config={"configurable": {"session_id": session_id}})

    # The response should be randomly generated.
    assert isinstance(response1, str)
    assert response1 != ""

    response2 = chain_with_history.invoke(
        {"input": "what was my previous message?"},
        config={"configurable": {"session_id": session_id}})

    # The response should be different from the input.
    assert response2 != ""

    assert len(history_store.store[session_id].messages) == 4