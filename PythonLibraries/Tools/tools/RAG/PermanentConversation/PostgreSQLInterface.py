from tools.Databases import (CommonSQLStatements, PostgreSQLConnection)
from .SQLStatements import SQLStatements
from .EmbedPermanentConversation import (
    ConversationMessageChunk,
)
from commonapi.Messages.PermanentConversation import PermanentConversation

from typing import List, Optional, Dict, Any
import asyncio

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
                success = await self._postgres_connection.create_extension("vector")
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
                await conn.execute(SQLStatements.CREATE_MESSAGE_PAIR_CHUNKS_INDEXES)
                
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

    async def get_chunks_by_conversation(self, conversation_id: str) -> List[ConversationMessageChunk]:
        """
        Get all chunks for a specific conversation.
        
        Args:
            conversation_id: The conversation ID to retrieve chunks for
            
        Returns:
            List of ConversationMessageChunk objects
        """
        try:
            async with self._postgres_connection.connect() as conn:
                rows = await conn.fetch(
                    SQLStatements.GET_CHUNKS_BY_CONVERSATION,
                    conversation_id)
                
                chunks = []
                for row in rows:
                    chunk = ConversationMessageChunk(
                        conversation_id=row['conversation_id'],
                        chunk_type=row['chunk_type'],
                        chunk_index=row['chunk_index'],
                        role=row['role'],
                        content=row['content'],
                        content_hash=row['content_hash'],
                        embedding=row['embedding']
                    )
                    chunks.append(chunk)
                return chunks
        except Exception as e:
            print(f"Error getting chunks by conversation: {e}")
            return []

    async def get_chunks_by_conversation_and_type(self, conversation_id: str, chunk_type: str) -> List[ConversationMessageChunk]:
        """
        Get chunks for a specific conversation and chunk type.
        
        Args:
            conversation_id: The conversation ID
            chunk_type: 'message' or 'message_pair'
            
        Returns:
            List of ConversationMessageChunk objects
        """
        try:
            async with self._postgres_connection.connect() as conn:
                rows = await conn.fetch(
                    SQLStatements.GET_CHUNKS_BY_CONVERSATION_AND_TYPE,
                    conversation_id, chunk_type)
                
                chunks = []
                for row in rows:
                    chunk = ConversationMessageChunk(
                        conversation_id=row['conversation_id'],
                        chunk_type=row['chunk_type'],
                        chunk_index=row['chunk_index'],
                        role=row['role'],
                        content=row['content'],
                        content_hash=row['content_hash'],
                        embedding=row['embedding']
                    )
                    chunks.append(chunk)
                return chunks
        except Exception as e:
            print(f"Error getting chunks by conversation and type: {e}")
            return []

    async def get_chunks_by_conversation_and_role(self, conversation_id: str, role: str) -> List[ConversationMessageChunk]:
        """
        Get chunks for a specific conversation and role.
        
        Args:
            conversation_id: The conversation ID
            role: 'user', 'assistant', 'system', or 'user_assistant'
            
        Returns:
            List of ConversationMessageChunk objects
        """
        try:
            async with self._postgres_connection.connect() as conn:
                rows = await conn.fetch(
                    SQLStatements.GET_CHUNKS_BY_CONVERSATION_AND_ROLE,
                    conversation_id, role)
                
                chunks = []
                for row in rows:
                    chunk = ConversationMessageChunk(
                        conversation_id=row['conversation_id'],
                        chunk_type=row['chunk_type'],
                        chunk_index=row['chunk_index'],
                        role=row['role'],
                        content=row['content'],
                        content_hash=row['content_hash'],
                        embedding=row['embedding']
                    )
                    chunks.append(chunk)
                return chunks
        except Exception as e:
            print(f"Error getting chunks by conversation and role: {e}")
            return []

    async def vector_similarity_search(self, query_embedding: List[float], conversation_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Perform vector similarity search within a specific conversation.
        
        Args:
            query_embedding: The query embedding vector
            conversation_id: The conversation ID to search within
            limit: Maximum number of results to return
            
        Returns:
            List of dictionaries containing chunk data and similarity scores
        """
        try:
            async with self._postgres_connection.connect() as conn:
                rows = await conn.fetch(
                    SQLStatements.VECTOR_SIMILARITY_SEARCH,
                    query_embedding, conversation_id, limit)
                
                results = []
                for row in rows:
                    result = {
                        'id': row['id'],
                        'conversation_id': row['conversation_id'],
                        'chunk_type': row['chunk_type'],
                        'chunk_index': row['chunk_index'],
                        'role': row['role'],
                        'content': row['content'],
                        'content_hash': row['content_hash'],
                        'embedding': row['embedding'],
                        'similarity_score': row['similarity_score'],
                        'created_at': row['created_at'],
                        'updated_at': row['updated_at']
                    }
                    results.append(result)
                return results
        except Exception as e:
            print(f"Error performing vector similarity search: {e}")
            return []

    async def vector_similarity_search_all(self, query_embedding: List[float], limit: int = 10) -> List[Dict[str, Any]]:
        """
        Perform vector similarity search across all conversations.
        
        Args:
            query_embedding: The query embedding vector
            limit: Maximum number of results to return
            
        Returns:
            List of dictionaries containing chunk data and similarity scores
        """
        try:
            async with self._postgres_connection.connect() as conn:
                rows = await conn.fetch(
                    SQLStatements.VECTOR_SIMILARITY_SEARCH_ALL,
                    query_embedding, limit)
                
                results = []
                for row in rows:
                    result = {
                        'id': row['id'],
                        'conversation_id': row['conversation_id'],
                        'chunk_type': row['chunk_type'],
                        'chunk_index': row['chunk_index'],
                        'role': row['role'],
                        'content': row['content'],
                        'content_hash': row['content_hash'],
                        'embedding': row['embedding'],
                        'similarity_score': row['similarity_score'],
                        'created_at': row['created_at'],
                        'updated_at': row['updated_at']
                    }
                    results.append(result)
                return results
        except Exception as e:
            print(f"Error performing vector similarity search all: {e}")
            return []

    async def vector_similarity_search_with_threshold(self, query_embedding: List[float], conversation_id: str, threshold: float, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Perform vector similarity search with a minimum similarity threshold.
        
        Args:
            query_embedding: The query embedding vector
            conversation_id: The conversation ID to search within
            threshold: Minimum similarity score (0.0 to 1.0)
            limit: Maximum number of results to return
            
        Returns:
            List of dictionaries containing chunk data and similarity scores
        """
        try:
            async with self._postgres_connection.connect() as conn:
                rows = await conn.fetch(
                    SQLStatements.VECTOR_SIMILARITY_SEARCH_WITH_THRESHOLD,
                    query_embedding, conversation_id, threshold, limit)
                
                results = []
                for row in rows:
                    result = {
                        'id': row['id'],
                        'conversation_id': row['conversation_id'],
                        'chunk_type': row['chunk_type'],
                        'chunk_index': row['chunk_index'],
                        'role': row['role'],
                        'content': row['content'],
                        'content_hash': row['content_hash'],
                        'embedding': row['embedding'],
                        'similarity_score': row['similarity_score'],
                        'created_at': row['created_at'],
                        'updated_at': row['updated_at']
                    }
                    results.append(result)
                return results
        except Exception as e:
            print(f"Error performing vector similarity search with threshold: {e}")
            return []

    async def get_chunk_by_id(self, chunk_id: int) -> Optional[ConversationMessageChunk]:
        """
        Get a specific chunk by its ID.
        
        Args:
            chunk_id: The chunk ID
            
        Returns:
            ConversationMessageChunk object or None if not found
        """
        try:
            async with self._postgres_connection.connect() as conn:
                row = await conn.fetchrow(
                    SQLStatements.GET_CHUNK_BY_ID,
                    chunk_id)
                
                if row:
                    return ConversationMessageChunk(
                        conversation_id=row['conversation_id'],
                        chunk_type=row['chunk_type'],
                        chunk_index=row['chunk_index'],
                        role=row['role'],
                        content=row['content'],
                        content_hash=row['content_hash'],
                        embedding=row['embedding']
                    )
                return None
        except Exception as e:
            print(f"Error getting chunk by ID: {e}")
            return None

    async def get_chunks_by_hash(self, content_hash: str) -> List[ConversationMessageChunk]:
        """
        Get all chunks with a specific content hash.
        
        Args:
            content_hash: The content hash to search for
            
        Returns:
            List of ConversationMessageChunk objects
        """
        try:
            async with self._postgres_connection.connect() as conn:
                rows = await conn.fetch(
                    SQLStatements.GET_CHUNKS_BY_HASH,
                    content_hash)
                
                chunks = []
                for row in rows:
                    chunk = ConversationMessageChunk(
                        conversation_id=row['conversation_id'],
                        chunk_type=row['chunk_type'],
                        chunk_index=row['chunk_index'],
                        role=row['role'],
                        content=row['content'],
                        content_hash=row['content_hash'],
                        embedding=row['embedding']
                    )
                    chunks.append(chunk)
                return chunks
        except Exception as e:
            print(f"Error getting chunks by hash: {e}")
            return []

    async def get_conversation_stats(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """
        Get statistics for a specific conversation.
        
        Args:
            conversation_id: The conversation ID
            
        Returns:
            Dictionary containing conversation statistics
        """
        try:
            async with self._postgres_connection.connect() as conn:
                row = await conn.fetchrow(
                    SQLStatements.GET_CONVERSATION_STATS,
                    conversation_id)
                
                if row:
                    return {
                        'conversation_id': row['conversation_id'],
                        'total_chunks': row['total_chunks'],
                        'message_chunks': row['message_chunks'],
                        'message_pair_chunks': row['message_pair_chunks'],
                        'user_chunks': row['user_chunks'],
                        'assistant_chunks': row['assistant_chunks'],
                        'system_chunks': row['system_chunks'],
                        'user_assistant_chunks': row['user_assistant_chunks'],
                        'first_chunk_created': row['first_chunk_created'],
                        'last_chunk_created': row['last_chunk_created']
                    }
                return None
        except Exception as e:
            print(f"Error getting conversation stats: {e}")
            return None

    async def conversation_exists(self, conversation_id: str) -> bool:
        """
        Check if a conversation exists in the database.
        
        Args:
            conversation_id: The conversation ID to check
            
        Returns:
            True if conversation exists, False otherwise
        """
        try:
            async with self._postgres_connection.connect() as conn:
                exists = await conn.fetchval(
                    SQLStatements.CONVERSATION_EXISTS,
                    conversation_id)
                return exists
        except Exception as e:
            print(f"Error checking if conversation exists: {e}")
            return False

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

    async def count_chunks_by_conversation(self, conversation_id: str) -> int:
        """
        Count the number of chunks for a specific conversation.
        
        Args:
            conversation_id: The conversation ID
            
        Returns:
            Number of chunks
        """
        try:
            async with self._postgres_connection.connect() as conn:
                count = await conn.fetchval(
                    SQLStatements.COUNT_CHUNKS_BY_CONVERSATION,
                    conversation_id)
                return count or 0
        except Exception as e:
            print(f"Error counting chunks by conversation: {e}")
            return 0

    async def get_recent_chunks(self, limit: int = 100) -> List[ConversationMessageChunk]:
        """
        Get the most recently created chunks.
        
        Args:
            limit: Maximum number of chunks to return
            
        Returns:
            List of ConversationMessageChunk objects
        """
        try:
            async with self._postgres_connection.connect() as conn:
                rows = await conn.fetch(
                    SQLStatements.GET_RECENT_CHUNKS,
                    limit)
                
                chunks = []
                for row in rows:
                    chunk = ConversationMessageChunk(
                        conversation_id=row['conversation_id'],
                        chunk_type=row['chunk_type'],
                        chunk_index=row['chunk_index'],
                        role=row['role'],
                        content=row['content'],
                        content_hash=row['content_hash'],
                        embedding=row['embedding']
                    )
                    chunks.append(chunk)
                return chunks
        except Exception as e:
            print(f"Error getting recent chunks: {e}")
            return []

    async def delete_chunks_by_conversation(self, conversation_id: str) -> bool:
        """
        Delete all chunks for a specific conversation.
        
        Args:
            conversation_id: The conversation ID
            
        Returns:
            True if successful, False otherwise
        """
        try:
            async with self._postgres_connection.connect() as conn:
                await conn.execute(
                    SQLStatements.DELETE_CHUNKS_BY_CONVERSATION,
                    conversation_id)
                return True
        except Exception as e:
            print(f"Error deleting chunks by conversation: {e}")
            return False

    async def delete_chunk_by_id(self, chunk_id: int) -> bool:
        """
        Delete a specific chunk by its ID.
        
        Args:
            chunk_id: The chunk ID
            
        Returns:
            True if successful, False otherwise
        """
        try:
            async with self._postgres_connection.connect() as conn:
                await conn.execute(
                    SQLStatements.DELETE_CHUNK_BY_ID,
                    chunk_id)
                return True
        except Exception as e:
            print(f"Error deleting chunk by ID: {e}")
            return False

    async def drop_tables(self) -> bool:
        """
        Drop the permanent conversation chunks table.
        """
        try:
            async with self._postgres_connection.connect() as conn:
                await conn.execute(f"DROP TABLE IF EXISTS {self.CHUNKS_TABLE_NAME}")
                return True
        except Exception as e:
            print(f"Error dropping table: {e}")
            return False

    # Legacy methods for backward compatibility (if needed)
    async def load_conversation_to_manager(self, conversation: PermanentConversation) -> bool:
        """
        Legacy method - not applicable for chunk-based system.
        Use get_chunks_by_conversation instead.
        """
        print("Warning: load_conversation_to_manager is not applicable for chunk-based system")
        return False

    async def save_conversation_from_manager(self, conversation: PermanentConversation) -> bool:
        """
        Legacy method - not applicable for chunk-based system.
        Use insert_chunks_batch instead.
        """
        print("Warning: save_conversation_from_manager is not applicable for chunk-based system")
        return False

