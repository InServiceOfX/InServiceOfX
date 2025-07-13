from commonapi.Databases import (CommonSQLStatements, PostgreSQLConnection)
from commonapi.Databases.Messages.PermanentConversation import SQLStatements

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

