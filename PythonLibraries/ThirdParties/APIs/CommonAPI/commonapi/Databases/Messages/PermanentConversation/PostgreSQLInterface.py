from commonapi.Databases import (CommonSQLStatements, PostgreSQLConnection)
from commonapi.Databases.Messages.PermanentConversation import SQLStatements
from commonapi.Messages.PermanentConversation import ConversationMessage

from typing import List, Optional

class PostgreSQLInterface:
    """Database interface for persisting permanent conversation messages."""

    TABLE_NAME = "permanent_conversation"

    def __init__(self, postgres_connection: PostgreSQLConnection):
        """
        Args:
            postgres_connection: PostgreSQLConnection instance
        """
        self._postgres_connection = postgres_connection
        self._database_name = postgres_connection._database_name

    async def create_table(self) -> bool:
        """
        Create the permanent_conversation table if it doesn't exist.
        """
        try:
            async with self._postgres_connection.connect() as conn:
                await conn.execute(SQLStatements.CREATE_PERMANENT_CONVERSATION_TABLE)
                return True
        except Exception as e:
            print(f"Error creating permanent_conversation table: {e}")

    async def table_exists(self) -> bool:
        """
        Check if the permanent_conversation table exists.
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
                    message.datetime,
                    message.role,
                    message.hash,
                    message.content,
                    message.embedding
                )
                return result
        except Exception as e:
            print(f"Error inserting permanent conversation message: {e}")
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

