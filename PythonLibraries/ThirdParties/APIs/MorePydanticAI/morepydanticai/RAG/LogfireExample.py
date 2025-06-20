from dataclasses import dataclass
from typing import List
from pydantic import TypeAdapter
from pydantic_ai import RunContext
from pydantic_ai.agent import Agent
from sentence_transformers import SentenceTransformer

import asyncpg
import asyncio
import pydantic_core
import re
import unicodedata

def slugify(value: str, separator: str, unicode: bool = False) -> str:
    """Slugify a string, to make it URL friendly."""
    # Taken unchanged from https://github.com/Python-Markdown/markdown/blob/3.7/markdown/extensions/toc.py#L38
    if not unicode:
        value = unicodedata.normalize('NFKD', value)
        value = value.encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value).strip().lower()
    return re.sub(rf'[{separator}\s]+', separator, value)

@dataclass
class DocsSection:
    """A section of the Pydantic Logfire Documentation."""
    id: int
    parent: int | None
    path: str
    level: int
    title: str
    content: str

    def url(self) -> str:
        url_path = re.sub(r'\.md$', '', self.path)
        return (
            f'https://logfire.pydantic.dev/docs/{url_path}/#{slugify(self.title, "-")}'
        )

    def embedding_content(self) -> str:
        return '\n\n'.join((
            f'path: {self.path}',
            f'title: {self.title}',
            self.content))

def create_type_adapter() -> TypeAdapter:
    """A type adapter provides type validation and serialization for types that
    aren't Pydantic models (in this case, a dataclass).    
    """
    return TypeAdapter(list[DocsSection])

async def insert_doc_section(
    sem: asyncio.Semaphore,
    embedding_model: SentenceTransformer,
    pool: asyncpg.Pool,
    section: DocsSection,
) -> None:
    """
    Ref:
    https://ai.pydantic.dev/examples/rag/#example-code
    """
    async with sem:
        url = section.url()
        exists = await pool.fetchval('SELECT 1 FROM doc_sections WHERE url = $1', url)
        if exists:
            return

        embedding = embedding_model.encode(
            section.embedding_content(),
            # Important for similarity search
            normalize_embeddings=True,
        )
        
        # embedding is now a numpy.ndarray, convert to list for JSON
        embedding_list = embedding.tolist()
        embedding_json = pydantic_core.to_json(embedding_list).decode()
        
        await pool.execute(
            'INSERT INTO doc_sections (url, title, content, embedding) VALUES ($1, $2, $3, $4)',
            url,
            section.title,
            section.content,
            embedding_json,
        )

async def build_search_database(
    embedding_model: SentenceTransformer,
    postgres_connection,
    database_schema: str,
    sections):
    """
    Ref:
    https://ai.pydantic.dev/examples/rag/#example-code
    """
    await postgres_connection.execute_with_pool(database_schema)

    # This then allows only 10 concurrent connections.
    sem = asyncio.Semaphore(10)
    async with asyncio.TaskGroup() as tg:
        for section in sections:
            tg.create_task(insert_doc_section(
                sem,
                embedding_model,
                postgres_connection._pool,
                section))

    return True
 
# Following is dedicated to the agent that does search or i.e. retrieval from
# the database.

@dataclass
class RagAgentDependencies:
    embedder: SentenceTransformer
    pool: asyncpg.Pool

def create_rag_agent(model: str) -> Agent:
    """
    Ref: 
    https://ai.pydantic.dev/api/agent/
    Agent dataclass
    Bases: Generic[AgentDepsT, OutputDataT]
    Agents are generic in the dependency type they take AgentDeptsT and the
    result data type they return, OutputDataT.
    """
    agent = Agent(model, deps_type=RagAgentDependencies)

    @agent.tool
    async def retrieve(
        context: RunContext[RagAgentDependencies],
        search_query: str) -> str:
        """Retrieve documentation sections based on a search query.
        Args:
            context: The call context.
            search_query: The search query.
        """
        embedding = context.deps.embedder.encode(
            search_query,
            normalize_embeddings=True)

        embedding_json = pydantic_core.to_json(embedding.tolist()).decode()

        rows = await context.deps.pool.fetch(
            'SELECT url, title, content FROM doc_sections ORDER BY embedding <-> $1 LIMIT 8',
            embedding_json,
        )
        return '\n\n'.join(
            f'# {row["title"]}\nDocumentation URL:{row["url"]}\n\n{row["content"]}\n'
            for row in rows
        )

    return agent

async def run_agent(
    agent: Agent,
    question: str,
    embedder: SentenceTransformer,
    pool: asyncpg.Pool):
    dependencies = RagAgentDependencies(embedder, pool)
    answer = await agent.run(question, deps=dependencies)
    #print(answer.output)
    return answer

async def run_agent_sequentially(
    model: str,
    questions: List[str],
    embedder: SentenceTransformer,
    pool: asyncpg.Pool,
    delay_seconds: float = 2.0):
    results = []

    dependencies = RagAgentDependencies(embedder, pool)

    for i, question in enumerate(questions):
        print(f"\n--- Processing question {i+1} of {len(questions)} ---")
        agent = create_rag_agent(model)
        answer = await agent.run(question, deps=dependencies)
        results.append(answer)

        if i < len(questions) - 1:
            print(
                f"\n--- Waiting {delay_seconds} seconds before next question ---")
            await asyncio.sleep(delay_seconds)

    return results
