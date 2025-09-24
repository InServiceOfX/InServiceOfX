from tools.Databases import (CommonSQLStatements, PostgreSQLConnection)
from .SQLStatements import SQLStatements
from .EmbedPermanentConversation import (
    ConversationMessageChunk,
)

from typing import List, Optional, Dict, Any
from warnings import warn
import asyncio, json

class PostgreSQLInterface:
    """Database interface for persisting permanent conversation chunks."""

    MESSAGE_CHUNKS_TABLE_NAME = "permanent_conversation_message_chunks"
    MESSAGE_PAIR_CHUNKS_TABLE_NAME = "permanent_conversation_message_pair_chunks"

    def __init__(self, postgres_connection: PostgreSQLConnection):
        """
        Args:
            postgres_connection: PostgreSQLConnection instance
        """
        self._postgres_connection = postgres_connection
        self._database_name = postgres_connection._database_name

    async def create_tables(self) -> bool:
        """
        Create the permanent conversation chunks tables and indexes if they don't exist.
        """
        try:
            # First, ensure pgvector extension is available
            if not await self._postgres_connection.extension_exists("vector"):
                success = await self._postgres_connection.create_extension(
                    "vector")
                if not success:
                    print("Failed to create pgvector extension")
                    return False
            
            async with self._postgres_connection.connect() as conn:
                # Create chunks table
                await conn.execute(
                    SQLStatements.CREATE_PERMANENT_CONVERSATION_MESSAGE_CHUNKS_TABLE)
                await conn.execute(
                    SQLStatements.CREATE_PERMANENT_CONVERSATION_MESSAGE_PAIR_CHUNKS_TABLE)

                # Create indexes for better performance
                await conn.execute(SQLStatements.CREATE_MESSAGE_CHUNKS_INDEXES)
                await conn.execute(
                    SQLStatements.CREATE_MESSAGE_PAIR_CHUNKS_INDEXES)
                
                return True
        except Exception as e:
            print(f"Error creating permanent conversation chunks tables: {e}")
            return False

    async def table_exists(self, table_name: str = None) -> bool:
        """
        Check if the chunks table exists.
        """
        try:
            target_table = table_name or self.MESSAGE_CHUNKS_TABLE_NAME
            async with self._postgres_connection.connect() as conn:
                exists = await conn.fetchval(
                    CommonSQLStatements.CHECK_TABLE_EXISTS,
                    target_table)
                return exists is not None
        except Exception as e:
            print(f"Error checking if table exists: {e}")
            return False

    async def insert_message_chunk(self, chunk: ConversationMessageChunk) \
        -> Optional[int]:
        """
        Insert a conversation message chunk into the database.
        
        Args:
            chunk: ConversationMessageChunk to insert
            
        Returns:
            The ID of the inserted chunk, or None if failed
        """
        try:
            async with self._postgres_connection.connect() as conn:
                result = await conn.fetchval(
                    SQLStatements.INSERT_MESSAGE_CHUNK,
                    chunk.conversation_id,
                    chunk.chunk_index,
                    chunk.total_chunks,
                    chunk.parent_message_hash,
                    chunk.content,
                    chunk.datetime,
                    chunk.hash,
                    chunk.role,
                    PostgreSQLConnection.convert_list_to_string(chunk.embedding)
                )
                return result
        except Exception as e:
            print(f"Error inserting message chunk: {e}")
            return None

    async def insert_message_pair_chunk(
            self,
            chunk: ConversationMessageChunk) -> Optional[int]:
        """
        Insert a conversation message pair chunk into the database.
        
        Args:
            chunk: ConversationMessagePairChunk to insert
            
        Returns:
            The ID of the inserted chunk, or None if failed
        """
        try:
            async with self._postgres_connection.connect() as conn:
                result = await conn.fetchval(
                    SQLStatements.INSERT_MESSAGE_PAIR_CHUNK,
                    chunk.conversation_id,
                    chunk.chunk_index,
                    chunk.total_chunks,
                    chunk.parent_message_hash,
                    chunk.content,
                    chunk.datetime,
                    chunk.hash,
                    chunk.role,
                    PostgreSQLConnection.convert_list_to_string(chunk.embedding)
                )
                return result
        except Exception as e:
            print(f"Error inserting message pair chunk: {e}")
            return None

    async def vector_similarity_search_message_chunks(
        self, 
        query_embedding: List[float], 
        role_filter: Optional[str] = None,
        similarity_threshold: Optional[float] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Perform vector similarity search on message chunks.
        
        Args:
            query_embedding: The query embedding vector
            role_filter: Optional role filter ('user', 'assistant', 'system')
            similarity_threshold: Optional minimum similarity score (0.0 to 1.0)
            limit: Maximum number of results to return
            
        Returns:
            List of dictionaries containing chunk data and similarity scores
        """
        try:
            async with self._postgres_connection.connect() as conn:
                query_embedding_as_string = \
                    PostgreSQLConnection.convert_list_to_string(query_embedding)
                rows = await conn.fetch(
                    SQLStatements.VECTOR_SIMILARITY_SEARCH_MESSAGE_CHUNKS,
                    query_embedding_as_string, role_filter, similarity_threshold, limit)
                
                results = []
                for row in rows:
                    result = {
                        'id': row['id'],
                        'conversation_id': row['conversation_id'],
                        'chunk_index': row['chunk_index'],
                        'total_chunks': row['total_chunks'],
                        'parent_message_hash': row['parent_message_hash'],
                        'content': row['content'],
                        'datetime': row['datetime'],
                        'hash': row['hash'],
                        'role': row['role'],
                        'embedding': row['embedding'],
                        'similarity_score': row['similarity_score'],
                        'created_at': row['created_at']
                    }
                    results.append(result)
                return results
        except Exception as e:
            print(f"Error performing vector similarity search on message chunks: {e}")
            return []

    async def get_all_conversation_ids(self) -> List[str]:
        """
        Get all unique conversation IDs from the database.
        
        Returns:
            List of conversation IDs
        """
        try:
            async with self._postgres_connection.connect() as conn:
                rows = await conn.fetch(SQLStatements.GET_ALL_CONVERSATION_IDS)
                return [row['conversation_id'] for row in rows]
        except Exception as e:
            print(f"Error getting all conversation IDs: {e}")
            return []

    async def drop_tables(self) -> bool:
        """
        Drop the permanent conversation chunks table.
        """
        try:
            async with self._postgres_connection.connect() as conn:
                await conn.execute(
                    f"DROP TABLE IF EXISTS {self.MESSAGE_CHUNKS_TABLE_NAME}")
                await conn.execute(
                    f"DROP TABLE IF EXISTS {self.MESSAGE_PAIR_CHUNKS_TABLE_NAME}")
                return True
        except Exception as e:
            print(f"Error dropping table: {e}")
            return False

    async def get_all_message_chunks(self) \
        -> List[ConversationMessageChunk]:
        """
        Retrieve all message chunks from the database. This function is
        typically used for testing.
        
        Returns:
            List of ConversationMessageChunk dataclass instances
        """
        try:
            async with self._postgres_connection.connect() as conn:
                rows = await conn.fetch(SQLStatements.GET_ALL_MESSAGE_CHUNKS)

                chunks = []
                for row in rows:
                    chunk = ConversationMessageChunk(
                        conversation_id=row['conversation_id'],
                        chunk_index=row['chunk_index'],
                        total_chunks=row['total_chunks'],
                        parent_message_hash=row['parent_message_hash'],
                        content=row['content'],
                        datetime=row['datetime'],
                        hash=row['hash'],
                        role=row['role'],
                        chunk_type="message",
                        embedding=json.loads(row['embedding']) \
                            if row['embedding'] is not None else None
                    )
                    chunks.append(chunk)
                return chunks
        except Exception as e:
            warn(f"Error retrieving all message chunks: {e}")
            return []

    async def get_all_message_pair_chunks(self) \
        -> List[ConversationMessageChunk]:
        """
        Retrieve all message pair chunks from the database. This function is
        typically used for testing.
        
        Returns:
            List of ConversationMessageChunk dataclass instances
        """
        try:
            async with self._postgres_connection.connect() as conn:
                rows = await conn.fetch(
                    SQLStatements.GET_ALL_MESSAGE_PAIR_CHUNKS)

                chunks = []
                for row in rows:
                    chunk = ConversationMessageChunk(
                        conversation_id=row['conversation_id'],
                        chunk_index=row['chunk_index'],
                        total_chunks=row['total_chunks'],
                        parent_message_hash=row['parent_message_hash'],
                        content=row['content'],
                        datetime=row['datetime'],
                        hash=row['hash'],
                        role=row['role'],
                        chunk_type="message_pair",
                        embedding=json.loads(row['embedding']) \
                            if row['embedding'] is not None else None
                    )
                    chunks.append(chunk)
                return chunks
        except Exception as e:
            warn(f"Error retrieving all message pair chunks: {e}")
            return []
