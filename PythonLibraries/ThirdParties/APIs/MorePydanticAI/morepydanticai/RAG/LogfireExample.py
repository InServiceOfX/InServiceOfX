from dataclasses import dataclass
from pydantic import TypeAdapter
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
            convert_to_numpy=True
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

