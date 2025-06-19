from dataclasses import dataclass
from sentence_transformers import SentenceTransformer

import asyncio
import asyncpg
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
    id: int
    parent: int | None
    path: str
    level: int
    title: str
    content: str

    def url(self) -> str:
        """Generate a URL-friendly identifier for the section."""
        url_path = re.sub(r'\.md$', '', self.path)
        return f'docs/{url_path}/#{slugify(self.title, "-")}'

    def embedding_content(self) -> str:
        """Content to embed: path, title, and content."""
        return '\n\n'.join((f'path: {self.path}', f'title: {self.title}', self.content))

async def insert_doc_section(
        sem: asyncio.Semaphore,
        embedder: SentenceTransformer,
        pool: asyncpg.Pool,
        section: DocsSection,
) -> None:
    """Insert a document section into the database with its embedding."""
    async with sem:
        url = section.url()
        exists = await pool.fetchval('SELECT 1 FROM doc_sections WHERE url = $1', url)
        if exists:
            logfire.info('Skipping {url=}', url=url)
            return

        with logfire.span('create embedding for {url=}', url=url):
            embedding = embedder.encode(
                [section.embedding_content()], normalize_embeddings=True
            )[0]
            embedding_json = pydantic_core.to_json(embedding.tolist()).decode()
        await pool.execute(
            'INSERT INTO doc_sections (url, title, content, embedding) VALUES ($1, $2, $3, $4)',
            url,
            section.title,
            section.content,
            embedding_json,
        )