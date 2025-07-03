"""
Database operations for system messages using PostgreSQL.
"""
from typing import List, Optional
from commonapi.Databases import (CommonSQLStatements, PostgreSQLConnection)
from commonapi.Databases.Messages.SystemMessages import SQLStatements
from commonapi.Messages import RecordedSystemMessage

class PostgreSQLInterface:
    """Database interface for persisting system messages."""
    
    TABLE_NAME = "system_messages"
    
    def __init__(self, postgres_connection: PostgreSQLConnection):
        """        
        Args:
            postgres_connection: PostgreSQLConnection instance
        """
        self._postgres_connection = postgres_connection
        self._database_name = postgres_connection._database_name
    
    async def create_table(self) -> bool:
        """
        Create the system_messages table if it doesn't exist.
        
        Returns:
            True if table was created or already exists, False otherwise
        """
        try:
            async with self._postgres_connection.connect() as conn:
                await conn.execute(SQLStatements.CREATE_SYSTEM_MESSAGES_TABLE)
                return True
        except Exception as e:
            print(f"Error creating system_messages table: {e}")
            return False
    
    async def table_exists(self) -> bool:
        """        
        Returns:
            True if table exists, False otherwise
        """
        try:
            async with self._postgres_connection.connect() as conn:
                exists = await conn.fetchval(
                    CommonSQLStatements.CHECK_TABLE_EXISTS,
                    self.TABLE_NAME)
                return exists is not None
        except Exception as e:
            print(f"Error checking if table exists: {e}")
            return False
    
    async def insert_message(self, message: RecordedSystemMessage) \
        -> Optional[int]:
        """
        Insert a system message into the database.
        Duplicates (based on hash) are ignored.
        
        Args:
            message: RecordedSystemMessage to insert
            
        Returns:
            ID of inserted message, or None if duplicate
        """
        try:
            async with self._postgres_connection.connect() as conn:
                result = await conn.fetchval(
                    SQLStatements.INSERT_SYSTEM_MESSAGE,
                    message.content,
                    message.timestamp,
                    message.hash
                )
                return result
        except Exception as e:
            print(f"Error inserting system message: {e}")
            return None
    
    async def load_all_messages(self) -> List[RecordedSystemMessage]:
        """        
        Returns:
            List of RecordedSystemMessage objects
        """
        try:
            async with self._postgres_connection.connect() as conn:
                rows = await conn.fetch(
                    SQLStatements.SELECT_ALL_SYSTEM_MESSAGES)
                
                messages = []
                for row in rows:
                    message = RecordedSystemMessage(
                        content=row['content'],
                        timestamp=row['timestamp'],
                        hash=row['hash'],
                        # Default to True, will be updated by SystemMessagesManager
                        is_active=True
                    )
                    messages.append(message)
                
                return messages
        except Exception as e:
            print(f"Error loading system messages: {e}")
            return []
    
    async def get_message_by_hash(self, hash_value: str) \
        -> Optional[RecordedSystemMessage]:
        """        
        Args:
            hash_value: Hash of the message to retrieve
            
        Returns:
            RecordedSystemMessage if found, None otherwise
        """
        try:
            async with self._postgres_connection.connect() as conn:
                row = await conn.fetchrow(
                    SQLStatements.SELECT_SYSTEM_MESSAGE_BY_HASH,
                    hash_value)
                
                if row:
                    return RecordedSystemMessage(
                        content=row['content'],
                        timestamp=row['timestamp'],
                        hash=row['hash'],
                        # Default to True, will be updated by SystemMessagesManager
                        is_active=True
                    )
                return None
        except Exception as e:
            print(f"Error getting message by hash: {e}")
            return None
    
    async def delete_message_by_hash(self, hash_value: str) -> bool:
        """        
        Args:
            hash_value: Hash of the message to delete
            
        Returns:
            True if message was deleted, False otherwise
        """
        try:
            async with self._postgres_connection.connect() as conn:
                result = await conn.fetchval(
                    SQLStatements.DELETE_SYSTEM_MESSAGE_BY_HASH,
                    hash_value)
                return result is not None
        except Exception as e:
            print(f"Error deleting message: {e}")
            return False
    
    async def load_messages_to_manager(self, manager) -> bool:
        """
        Load all messages from database and add them to a SystemMessagesManager.
        
        Args:
            manager: SystemMessagesManager instance to populate
            
        Returns:
            True if successful, False otherwise
        """
        try:
            messages = await self.load_all_messages()
            for message in messages:
                manager.add_previously_recorded_message(message)
            return True
        except Exception as e:
            print(f"Error saving messages to manager: {e}")
            return False
    
    async def drop_table(self) -> bool:
        """
        Drop the system_messages table.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            async with self._postgres_connection.connect() as conn:
                await conn.execute(SQLStatements.DROP_SYSTEM_MESSAGES_TABLE)
                return True
        except Exception as e:
            print(f"Error dropping table: {e}")
            return False
