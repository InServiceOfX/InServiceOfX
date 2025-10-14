"""
USAGE:
You'll also want to monitor real-time changes to the PostgreSQL database server
you are running with Docker. Follow the instructions in the README.md of this
(sub-)project but otherwise, this is a summary:

Get the name and container ID of the Docker image that has postgresql running
docker ps
`exec` into the running Docker container, e.g.
docker exec -it local-llm-full-postgres psql -U inserviceofx -d local_llm_full_database
where you'll get the username and database name from the docker-compose.yml file.

You should see something like this:
psql (16.8 (Debian 16.8-1.pgdg120+1))
Type "help" for help.

Then you can do something like this:

# list all databases
\l
"""
from TestSetup.setup_PermanentConversation_RAG \
    import (
        setup_PermanentConversation_RAG_dependences,
        setup_PermanentConversation_RAG
        )

model_path, is_model_downloaded, model_is_not_downloaded_message, \
    test_db_name, load_test_conversation, cleanup_test_database, \
    postgres_connection, test_dsn_value = \
        setup_PermanentConversation_RAG_dependences()

from tools.Databases import PostgreSQLConnection
from tools.RAG.PermanentConversation import (
    PostgreSQLInterface,
    RAGProcessor,
    RAGTool,
    )

import pytest
import pytest_asyncio
import time
import torch

@pytest_asyncio.fixture(scope="function")
def test_db_name():
    return "test_permanent_conversation_database"

@pytest_asyncio.fixture(scope="session")
def test_dsn():
    return test_dsn_value

test_queries = [
    "HTML CSS JavaScript ball hexagon animation physics gravity",
    "Go language recursive function maze solving algorithm", 
    "transformer neural network attention mechanism machine learning"
]

@pytest.mark.asyncio
async def test_PermanentConversation_RAGTool_with_conversation(
    test_dsn: str,
    test_db_name: str,
    postgres_connection: PostgreSQLConnection):

    pgsql_interface, embed_pc = \
        await setup_PermanentConversation_RAG(
            postgres_connection,
            test_db_name,
            model_path,
            load_test_conversation
        )

    rag_processor = RAGProcessor(
        pgsql_interface,
        embed_pc)

    rag_tool = RAGTool(rag_processor)

    results = []

    for query in test_queries:
        result = await rag_tool.retrieve_context(
            query=query,
            max_chunks=7,
            is_return_matches=True)
        results.append(result)

    for result in results:
        assert "message" in result, "message should be in result"
        assert "context" in result, "context should be in result"
        assert "search_results" in result, "search_results should be in result"
        assert result["message"] is not None, "message should not be None"
        assert result["context"] is not None, "context should not be None"
        assert result["search_results"] is not None, \
            "search_results should not be None"

    for index, result in enumerate(results):
        print(f"Query {index}: {test_queries[index]}")
        print(f"Result message {index}: {result['message'][:100]}")
        print(f"Result context {index}: {result['context'][:100]}")
        print("--------------------------------")

    # For each query, there are multiple search results, for each chunk of the
    # query itself (the query itself gets split into chunks).
    print(results[0]["search_results"][0][0].keys())

@pytest.mark.asyncio
async def test_PermanentConversation_RAGTool_with_conversation(
    test_dsn: str,
    test_db_name: str,
    postgres_connection: PostgreSQLConnection):

    pgsql_interface, embed_pc = \
        await setup_PermanentConversation_RAG(
            postgres_connection,
            test_db_name,
            model_path,
            load_test_conversation
        )

    rag_processor = RAGProcessor(
        pgsql_interface,
        embed_pc)

    rag_tool = RAGTool(rag_processor)

    results = []

    for query in test_queries:
        result = await rag_tool.retrieve_context(
            query=query,
            max_chunks=7,
            is_return_matches=True)
        results.append(result)

    for result in results:
        assert "message" in result, "message should be in result"
        assert "context" in result, "context should be in result"
        assert "search_results" in result, "search_results should be in result"
        assert result["message"] is not None, "message should not be None"
        assert result["context"] is not None, "context should not be None"
        assert result["search_results"] is not None, \
            "search_results should not be None"

    for index, result in enumerate(results):
        print(f"Query {index}: {test_queries[index]}")
        print(f"Result message {index}: {result['message'][:100]}")
        print(f"Result context {index}: {result['context'][:100]}")
        print("--------------------------------")

    # For each query, there are multiple search results, for each chunk of the
    # query itself (the query itself gets split into chunks).
    print(results[0]["search_results"][0][0].keys())

from corecode.Utilities import DataSubdirectories, is_model_there
data_subdirectories = DataSubdirectories()
relative_llm_model_path = "Models/LLM/Qwen/Qwen3-0.6B"
is_llm_model_downloaded, llm_model_path = is_model_there(
    relative_llm_model_path,
    data_subdirectories)
llm_model_is_not_downloaded_message = \
    f"Model {relative_llm_model_path} not downloaded"

from commonapi.Messages import (
    AssistantMessage,
    ConversationSystemAndPermanent,
    UserMessage)

from moretransformers.Applications import ModelAndTokenizer
from moretransformers.Configurations import (
    CreateDefaultGenerationConfigurations,
    FromPretrainedModelConfiguration,
    FromPretrainedTokenizerConfiguration,)
from moretransformers.Tools import ToolCallProcessor
from tools.Managers import ModelAndToolCallManager

def setup_tests_with_ModelAndToolCallManager():
    csp = ConversationSystemAndPermanent()
    from_pretrained_tokenizer_configuration = FromPretrainedTokenizerConfiguration(
        pretrained_model_name_or_path=llm_model_path)
    from_pretrained_model_configuration = FromPretrainedModelConfiguration(
        pretrained_model_name_or_path=llm_model_path,
        device_map="cuda:0",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2")
    generation_configuration = \
        CreateDefaultGenerationConfigurations.for_Qwen3_thinking()
    generation_configuration.do_sample = True
    mat = ModelAndTokenizer(
        llm_model_path,
        from_pretrained_model_configuration=from_pretrained_model_configuration,
        from_pretrained_tokenizer_configuration=\
            from_pretrained_tokenizer_configuration,
        generation_configuration=generation_configuration)
    return mat, csp

@pytest.mark.asyncio
async def test_PermanentConversation_RAGTool_with_ModelAndToolCallManager(
    test_dsn: str,
    test_db_name: str,
    postgres_connection: PostgreSQLConnection):

    pgsql_interface, embed_pc = \
        await setup_PermanentConversation_RAG(
            postgres_connection,
            test_db_name,
            model_path,
            load_test_conversation
        )

    rag_processor = RAGProcessor(
        pgsql_interface,
        embed_pc)

    rag_tool = RAGTool(rag_processor)

    mat, csp = setup_tests_with_ModelAndToolCallManager()