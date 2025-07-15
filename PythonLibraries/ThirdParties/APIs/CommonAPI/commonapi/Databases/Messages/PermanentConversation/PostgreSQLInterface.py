from commonapi.Databases import (CommonSQLStatements, PostgreSQLConnection)
from commonapi.Databases.Messages.PermanentConversation import SQLStatements
from commonapi.Messages.PermanentConversation import (
    ConversationMessage, 
    ConversationMessagePair,
    PermanentConversation
)

from typing import List, Optional, Tuple

class PostgreSQLInterface:
    """Database interface for persisting permanent conversation messages and
    pairs."""

    MESSAGES_TABLE_NAME = "permanent_conversation_messages"
    MESSAGE_PAIRS_TABLE_NAME = "permanent_conversation_message_pairs"

    def __init__(self, postgres_connection: PostgreSQLConnection):
        """
        Args:
            postgres_connection: PostgreSQLConnection instance
        """
        self._postgres_connection = postgres_connection
        self._database_name = postgres_connection._database_name

    async def create_tables(self) -> bool:
        """
        Create both permanent conversation tables if they don't exist.
        """
        try:
            # First, ensure pgvector extension is available
            if not await self._postgres_connection.extension_exists("vector"):
                success = \
                    await self._postgres_connection.create_extension("vector")
                if not success:
                    print("Failed to create pgvector extension")
                    return False
            
            async with self._postgres_connection.connect() as conn:
                # Create messages table
                await conn.execute(
                    SQLStatements.CREATE_PERMANENT_CONVERSATION_MESSAGES_TABLE)
                
                # Create message pairs table
                await conn.execute(SQLStatements.CREATE_PERMANENT_CONVERSATION_MESSAGE_PAIRS_TABLE)
                
                return True
        except Exception as e:
            print(f"Error creating permanent conversation tables: {e}")
            return False

    async def table_exists(self, table_name: str = None) -> bool:
        """
        Check if the specified table exists.
        """
        try:
            target_table = table_name or self.MESSAGES_TABLE_NAME
            async with self._postgres_connection.connect() as conn:
                exists = await conn.fetchval(
                    CommonSQLStatements.CHECK_TABLE_EXISTS,
                    target_table)
                return exists is not None
        except Exception as e:
            print(f"Error checking if table exists: {e}")
            return False

    async def insert_message(self, message: ConversationMessage) \
        -> Optional[int]:
        """
        Insert a conversation message into the database.
        Duplicates (based on hash) are ignored.
        """
        try:
            async with self._postgres_connection.connect() as conn:
                result = await conn.fetchval(
                    SQLStatements.INSERT_PERMANENT_CONVERSATION_MESSAGE,
                    message.conversation_id,
                    message.content,
                    message.datetime,
                    message.hash,
                    message.role,
                    message.embedding
                )
                return result
        except Exception as e:
            print(f"Error inserting permanent conversation message: {e}")
            return None

    async def insert_message_pair(self, message_pair: ConversationMessagePair) \
        -> Optional[int]:
        """
        Insert a conversation message pair into the database.
        Duplicates (based on hash) are ignored.
        """
        try:
            async with self._postgres_connection.connect() as conn:
                result = await conn.fetchval(
                    SQLStatements.INSERT_PERMANENT_CONVERSATION_MESSAGE_PAIR,
                    message_pair.conversation_pair_id,
                    message_pair.content_0,
                    message_pair.content_1,
                    message_pair.datetime,
                    message_pair.hash,
                    message_pair.role_0,
                    message_pair.role_1,
                    message_pair.embedding
                )
                return result
        except Exception as e:
            print(f"Error inserting permanent conversation message pair: {e}")
            return None

    async def get_all_messages(self) -> List[ConversationMessage]:
        try:
            async with self._postgres_connection.connect() as conn:
                rows = await conn.fetch(
                    SQLStatements.SELECT_ALL_PERMANENT_CONVERSATION_MESSAGES)
                
                messages = []
                for row in rows:
                    message = ConversationMessage(
                        conversation_id=row['conversation_id'],
                        content=row['content'],
                        datetime=row['datetime'],
                        hash=row['hash'],
                        role=row['role'],
                        embedding=row['embedding']
                    )
                    messages.append(message)
                return messages
        except Exception as e:
            print(f"Error getting all permanent conversation messages: {e}")
            return []

    async def get_all_message_pairs(self) -> List[ConversationMessagePair]:
        """
        Get all conversation message pairs from the database.
        """
        try:
            async with self._postgres_connection.connect() as conn:
                rows = await conn.fetch(
                    SQLStatements.SELECT_ALL_PERMANENT_CONVERSATION_MESSAGE_PAIRS)
                
                message_pairs = []
                for row in rows:
                    message_pair = ConversationMessagePair(
                        conversation_pair_id=row['conversation_pair_id'],
                        content_0=row['content_0'],
                        content_1=row['content_1'],
                        datetime=row['datetime'],
                        hash=row['hash'],
                        role_0=row['role_0'],
                        role_1=row['role_1'],
                        embedding=row['embedding']
                    )
                    message_pairs.append(message_pair)
                return message_pairs
        except Exception as e:
            print(f"Error getting all permanent conversation message pairs: {e}")
            return []

    async def get_message_by_hash(self, hash_value: str) -> Optional[ConversationMessage]:
        """
        Get a conversation message by its hash.
        """
        try:
            async with self._postgres_connection.connect() as conn:
                row = await conn.fetchrow(
                    SQLStatements.SELECT_PERMANENT_CONVERSATION_MESSAGE_BY_HASH,
                    hash_value)
                
                if row:
                    return ConversationMessage(
                        conversation_id=row['conversation_id'],
                        content=row['content'],
                        datetime=row['datetime'],
                        hash=row['hash'],
                        role=row['role'],
                        embedding=row['embedding']
                    )
                return None
        except Exception as e:
            print(f"Error getting message by hash: {e}")
            return None

    async def get_message_pair_by_hash(self, hash_value: str) -> Optional[ConversationMessagePair]:
        """
        Get a conversation message pair by its hash.
        """
        try:
            async with self._postgres_connection.connect() as conn:
                row = await conn.fetchrow(
                    SQLStatements.SELECT_PERMANENT_CONVERSATION_MESSAGE_PAIR_BY_HASH,
                    hash_value)
                
                if row:
                    return ConversationMessagePair(
                        conversation_pair_id=row['conversation_pair_id'],
                        content_0=row['content_0'],
                        content_1=row['content_1'],
                        datetime=row['datetime'],
                        hash=row['hash'],
                        role_0=row['role_0'],
                        role_1=row['role_1'],
                        embedding=row['embedding']
                    )
                return None
        except Exception as e:
            print(f"Error getting message pair by hash: {e}")
            return None

    async def _delete_message_by_hash(self, hash_value: str) -> bool:
        """We don't expect the user to delete messages from the permanent
        conversation."""
        try:
            async with self._postgres_connection.connect() as conn:
                result = await conn.fetchval(
                    SQLStatements.DELETE_PERMANENT_CONVERSATION_MESSAGE_BY_HASH,
                    hash_value)
                return result is not None
        except Exception as e:
            print(f"Error deleting message: {e}")
            return False

    async def _delete_message_pair_by_hash(self, hash_value: str) -> bool:
        """We don't expect the user to delete message pairs from the permanent
        conversation."""
        try:
            async with self._postgres_connection.connect() as conn:
                result = await conn.fetchval(
                    SQLStatements.DELETE_PERMANENT_CONVERSATION_MESSAGE_PAIR_BY_HASH,
                    hash_value)
                return result is not None
        except Exception as e:
            print(f"Error deleting message pair: {e}")
            return False

    async def load_conversation_to_manager(
            self,
            conversation: PermanentConversation) -> bool:
        """
        Load all messages and message pairs from database into a
        PermanentConversation.

        Args:
            conversation: PermanentConversation instance to populate
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load messages
            messages = await self.get_all_messages()
            for message in messages:
                conversation.messages.append(message)
                conversation.content_hashes.append(message.hash)
                if message.hash not in conversation.hash_to_indices_reverse_map:
                    conversation.hash_to_indices_reverse_map[message.hash] = \
                        [len(conversation.content_hashes) - 1,]
                else:
                    conversation.hash_to_indices_reverse_map[message.hash].append(
                        len(conversation.content_hashes) - 1)
            
            # Load message pairs
            message_pairs = await self.get_all_message_pairs()
            for message_pair in message_pairs:
                conversation.message_pairs.append(message_pair)
            
            # Update counters
            if messages:
                conversation._counter = max(msg.conversation_id for msg in messages) + 1
            if message_pairs:
                conversation._message_pair_counter = max(pair.conversation_pair_id for pair in message_pairs) + 1
            
            return True
        except Exception as e:
            print(f"Error loading conversation to manager: {e}")
            return False

    async def save_conversation_from_manager(
            self,
            conversation: PermanentConversation) -> bool:
        """
        Save all messages and message pairs from a PermanentConversation to
        database.
        
        Args:
            conversation: PermanentConversation instance to save
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Save messages
            for message in conversation.messages:
                await self.insert_message(message)
            
            # Save message pairs
            for message_pair in conversation.message_pairs:
                await self.insert_message_pair(message_pair)
            
            return True
        except Exception as e:
            print(f"Error saving conversation from manager: {e}")
            return False

    async def drop_tables(self) -> bool:
        """
        Drop both permanent conversation tables.
        """
        try:
            async with self._postgres_connection.connect() as conn:
                await conn.execute(
                    SQLStatements.DROP_PERMANENT_CONVERSATION_MESSAGE_PAIRS_TABLE)
                await conn.execute(
                    SQLStatements.DROP_PERMANENT_CONVERSATION_MESSAGES_TABLE)
                return True
        except Exception as e:
            print(f"Error dropping tables: {e}")
            return False

    async def get_max_ids(self) -> Tuple[int, int]:
        """
        Get the maximum conversation_id and conversation_pair_id from the
        database.
        """
        try:
            async with self._postgres_connection.connect() as conn:
                max_conversation_id = await conn.fetchval(
                    SQLStatements.GET_MAX_CONVERSATION_ID)
                max_conversation_pair_id = await conn.fetchval(
                    SQLStatements.GET_MAX_CONVERSATION_PAIR_ID)
                return max_conversation_id, max_conversation_pair_id
        except Exception as e:
            print(f"Error getting max IDs: {e}")
            return -1, -1

