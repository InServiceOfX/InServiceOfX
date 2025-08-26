from commonapi.Databases import (CommonSQLStatements, PostgreSQLConnection)
from commonapi.Databases.RAG.TextPDF import SQLStatements
from warnings import warn

class PostgreSQLInterface:
    """Database interface for persisting text PDFs and their chunks."""

    TEXT_PDFS_TABLE_NAME = "text_pdfs"
    TEXT_PDF_CHUNKS_TABLE_NAME = "text_pdf_chunks"

    def __init__(self, postgres_connection: PostgreSQLConnection):
        self._postgres_connection = postgres_connection
        self._database_name = postgres_connection._database_name

    async def create_text_pdfs_table(self) -> bool:
        try:
            # First, ensure pgvector extension is available
            if not await self._postgres_connection.extension_exists("vector"):
                success = \
                    await self._postgres_connection.create_extension("vector")
                if not success:
                    print("Failed to create pgvector extension")
                    return False

            async with self._postgres_connection.connect() as conn:
                # Create text_pdfs table
                await conn.execute(SQLStatements.CREATE_TEXT_PDFS_TABLE)

                return True
        except Exception as e:
            warn(f"Error creating text_pdfs table: {e}")
            return False

    async def create_text_pdf_chunks_table(self) -> bool:
        try:
            async with self._postgres_connection.connect() as conn:
                # Create text_pdf_chunks table
                await conn.execute(SQLStatements.CREATE_TEXT_PDF_CHUNKS_TABLE)

                return True
        except Exception as e:
            warn(f"Error creating text_pdf_chunks table: {e}")
            return False

    async def table_exists(self, table_name: str = None) -> bool:
        """
        Check if the specified table exists.
        """
        try:
            target_table = table_name or self.TEXT_PDFS_TABLE_NAME
            async with self._postgres_connection.connect() as conn:
                exists = await conn.fetchval(
                    CommonSQLStatements.CHECK_TABLE_EXISTS,
                    target_table)
                return exists is not None
        except Exception as e:
            warn(f"Error checking if table exists: {e}")
            return False

    async def create_index_for_similarity_search(self) -> bool:
        try:
            async with self._postgres_connection.connect() as conn:
                await conn.execute(SQLStatements.CREATE_INDEX_FOR_SIMILARITY_SEARCH)
                return True
        except Exception as e:
            warn(f"Error creating index for similarity search: {e}")
            return False

    async def insert_text_pdf(self, filename: str, total_chunks: int) \
          -> Optional[int]:
        """Insert a text PDF record."""
        try:
            async with self._postgres_connection.connect() as conn:
                result = await conn.fetchval(
                    SQLStatements.INSERT_TEXT_PDF,
                    filename,
                    total_chunks)
                return result
        except Exception as e:
            warn(f"Error inserting text PDF: {e}")
            return None

    async def insert_text_pdf_chunk(
            self,
            document_id: int,
            chunk_id: int,
            content: str,
            page_number: int,
            chunk_hash: str,
            embedding: list,
            metadata: dict) -> Optional[int]:
        """Insert a text PDF chunk record."""
        try:
            async with self._postgres_connection.connect() as conn:
                result = await conn.fetchval(
                    SQLStatements.INSERT_TEXT_PDF_CHUNK,
                    document_id,
                    chunk_id,
                    content,
                    page_number,
                    chunk_hash,
                    embedding,
                    metadata)
                return result
        except Exception as e:
            warn(f"Error inserting text PDF chunk: {e}")
            return None

    async def search_similar_chunks(self, embedding: list, limit: int = 5) \
        -> List[Dict[str, Any]]:
        """Search for similar chunks using the embedding."""
        try:
            async with self._postgres_connection.connect() as conn:
                rows = await conn.fetch(
                    SQLStatements.SELECT_TEXT_PDF_CHUNKS_BY_EMBEDDING,
                    embedding,
                    limit)
                results = []
                for row in rows:
                    results.append({
                        'id': row['id'],
                        'content': row['content'],
                        'page_number': row['page_number'],
                        'metadata': row['metadata'],
                        'filename': row['filename'],
                        'distance': row['distance']
                    })
                return results
        except Exception as e:
            warn(f"Error searching similar chunks: {e}")
            return []

    async def get_text_pdf_chunks_by_filename(self, filename: str) \
        -> List[Dict[str, Any]]:
        """Get all chunks for a specific text PDF."""
        try:
            async with self._postgres_connection.connect() as conn:
                rows = await conn.fetch(
                    SQLStatements.SELECT_TEXT_PDF_CHUNKS_BY_FILENAME,
                    filename)
                results = []
                for row in rows:
                    results.append({
                        'chunk_id': row['chunk_id'],
                        'content': row['content'],
                        'page_number': row['page_number'],
                        'metadata': row['metadata']
                    })
                return results
        except Exception as e:
            warn(f"Error getting text PDF chunks by filename: {e}")
            return []
