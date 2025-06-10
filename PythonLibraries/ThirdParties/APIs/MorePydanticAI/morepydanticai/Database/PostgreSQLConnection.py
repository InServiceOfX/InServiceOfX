from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Optional
import asyncpg

class PostgreSQLConnection:
    def __init__(self, server_data_source_name: str, database_name: str):
        self._server_data_source_name = server_data_source_name
        self._database_name = database_name
        self._connection: asyncpg.Connection | None = None

    @asynccontextmanager
    async def connect(self, database_name: Optional[str] = None) -> \
        AsyncGenerator[Any, None]:
        if database_name is None:
            database_name = self._database_name
        else:
            self._database_name = database_name

        if self._connection is None:
            self._connection = await asyncpg.connect(
                self._server_data_source_name)
            try:
                database_exists = await self._connection.fetchval(
                    # pg_database is a PostgreSQL's internal system tables
                    # (system catalogs)
                    f'SELECT 1 FROM pg_database WHERE datname = $1',
                    database_name
                )
                if not database_exists:
                    yield None
                    return

            finally:
                await self._connection.close()

            self._connection = await asyncpg.connect(
                f"{self._server_data_source_name}/{database_name}"
            )

        try:
            # yield creates a generator function, pauses function execution and
            # returns a value; when function called again, it resumes from where
            # it left off.
            yield self._connection
        except Exception as error:
            if self._connection:
                await self._connection.close()
                self._connection = None
            raise error

    async def create_database(self, database_name: str):
        print(f"Attempting to create database: {database_name}")
        self._database_name = database_name

        try:
            print(f"Connecting to server with DSN: {self._server_data_source_name}")
            connection = await asyncpg.connect(self._server_data_source_name)
            print("Successfully connected to server")
        except Exception as e:
            print(f"Failed to connect to server: {str(e)}")
            raise

        try:
            print("Checking if database exists...")
            database_exists = await connection.fetchval(
                'SELECT 1 FROM pg_database WHERE datname = $1',
                database_name
            )

            if not database_exists:
                print(f"Creating database {database_name}...")
                try:
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
                    
                    # Verify the database was created
                    created = await connection.fetchval(
                        'SELECT 1 FROM pg_database WHERE datname = $1',
                        database_name
                    )
                    print(f"Database creation verification: {created is not None}")

                except Exception as e:
                    print(f"Error during database creation: {str(e)}")
                    raise

        except Exception as e:
            print(f"Error in database operations: {str(e)}")
            raise
        finally:
            print("Closing connection to server")
            await connection.close()

        print("Creating connection to new database...")
        try:
            # Connect to the new database
            new_dsn = f"{self._server_data_source_name}/{database_name}"
            print(f"Connecting to new database with DSN: {new_dsn}")
            self._connection = await asyncpg.connect(new_dsn)
            print("Successfully connected to new database")
        except Exception as e:
            print(f"Error connecting to new database: {str(e)}")
            raise

    async def close(self):
        await self._connection.close()

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
