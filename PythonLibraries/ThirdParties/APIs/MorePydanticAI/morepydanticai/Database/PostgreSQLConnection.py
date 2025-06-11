from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Optional
import asyncpg

class PostgreSQLConnection:
    # For PostgreSQL, the system database is called "postgres"
    DEFAULT_SYSTEM_DB = "postgres"

    def __init__(self, server_data_source_name: str, database_name: str):
        """
        Args:
            server_data_source_name: The DSN without database name (e.g.,
                "postgresql://user:pass@host:port")
            database_name: The database to connect to
        """
        # Ensure the DSN doesn't include a database name
        self._server_data_source_name = server_data_source_name
        self._database_name = database_name
        self._connection: asyncpg.Connection | None = None

    @property
    def system_dsn(self) -> str:
        """Get the DSN for connecting to the system database."""
        return f"{self._server_data_source_name}/{self.DEFAULT_SYSTEM_DB}"

    async def list_all_databases(self) -> list[str]:
        """List all non-template databases by connecting to the system database."""
        try:
            # Connect to the system database
            conn = await asyncpg.connect(self.system_dsn)
            try:
                rows = await conn.fetch(
                    "SELECT datname FROM pg_database WHERE datistemplate = false ORDER BY datname;"
                )            
                return [row["datname"] for row in rows]
            finally:
                await conn.close()
        except Exception as e:
            print(f"Error listing databases: {e}")
            raise

    async def database_exists(self, database_name: str) -> bool:
        try:
            conn = await asyncpg.connect(self.system_dsn)
            # pg_database is a PostgreSQL's internal system tables (system
            # catalogs)
            exists = await conn.fetchval(
                'SELECT 1 FROM pg_database WHERE datname = $1',
                database_name
            )
            return exists is not None
        except Exception as e:
            print(f"Error checking if database exists: {e}")
            raise

    @asynccontextmanager
    async def connect(self, database_name: Optional[str] = None) -> \
        AsyncGenerator[Any, None]:
        if database_name is None and self._database_name is None:
            raise ValueError(
                "No database name provided and no default database name set")
        elif database_name is None:
            database_name = self._database_name
        else:
            self._database_name = database_name

        database_exists = await self.database_exists(database_name)

        if not database_exists:
            raise ValueError(
                f"Database {database_name} does not exist")

        if self._connection is not None:
            if not self._connection.closed:
                await self._connection.close()
            self._connection = None

        self._connection = await asyncpg.connect(
            f"{self._server_data_source_name}/{database_name}")

        yield self._connection

    async def create_database(self, database_name: str):
        database_exists = await self.database_exists(database_name)
        if database_exists:
            print(f"Database {database_name} already exists")
            return

        self._database_name = database_name

        connection = await asyncpg.connect(self.system_dsn)

        try:
            print(f"Creating database {database_name}...")
            # First check if we have permission
            is_superuser = await connection.fetchval("""
                SELECT usesuper FROM pg_user WHERE usename = current_user
            """)
            print(f"Current user is superuser: {is_superuser}")
                    
            if not is_superuser:
                can_create = await connection.fetchval("""
                    SELECT has_database_privilege(current_user, 'CREATE')
                """)
                print(f"Current user can create databases: {can_create}")
                if not can_create:
                    raise PermissionError(
                        "Current user does not have permission to create databases")

            # Try to create the database
            await connection.execute(
                f'CREATE DATABASE {database_name}'
            )
            print("Database creation command executed successfully")
                    
        except Exception as e:
            print(f"Error in database operations: {str(e)}")
            raise
        finally:
            print("Closing connection to server")
            await connection.close()

    async def close(self):
        if self._connection is not None:
            if not self._connection.closed:
                await self._connection.close()
            self._connection = None

    async def execute(self, query: str, *args):
        return await self._connection.execute(query, *args)

    async def fetch_value(
            self,
            query: str,
            database_name: Optional[str] = None,
            *args) -> Any:
        """
        Execute a query and return a single value.
        
        Args:
            query: SQL query to execute
            database_name: Optional database name to connect to
            *args: Query parameters
            
        Returns:
            Single value from the query result
        """
        async with self.connect(database_name) as conn:
            if conn is None:
                raise ValueError(
                    f"Database {database_name or self._database_name} does not exist")
            return await conn.fetchval(query, *args)

    async def fetch_row(
            self,
            query: str,
            database_name: Optional[str] = None,
            *args) -> Optional[dict]:
        """
        Execute a query and return a single row.
        
        Args:
            query: SQL query to execute
            database_name: Optional database name to connect to
            *args: Query parameters
            
        Returns:
            Single row as a dictionary or None if no rows found
        """
        async with self.connect(database_name) as conn:
            if conn is None:
                raise ValueError(
                    f"Database {database_name or self._database_name} does not exist")
            return await conn.fetchrow(query, *args)

    async def fetch_all(
            self,
            query: str,
            database_name: Optional[str] = None,
            *args) -> list[dict]:
        """
        Execute a query and return all rows.
        
        Args:
            query: SQL query to execute
            database_name: Optional database name to connect to
            *args: Query parameters
            
        Returns:
            List of rows as dictionaries
        """
        async with self.connect(database_name) as conn:
            if conn is None:
                raise ValueError(
                    f"Database {database_name or self._database_name} does not exist")
            return await conn.fetch(query, *args)
