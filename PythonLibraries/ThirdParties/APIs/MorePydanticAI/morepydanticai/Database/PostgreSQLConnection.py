from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Optional, List, Dict
import asyncpg

class PostgreSQLConnection:
    # For PostgreSQL, the system database is called "postgres"
    DEFAULT_SYSTEM_DB = "postgres"

    def __init__(self, server_data_source_name: str, database_name: str = None):
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
        # https://magicstack.github.io/asyncpg/current/usage.html
        # For server-type applications that handle frequent requests and need
        # the database connection for a short period time while handling a
        # request, use of a connection pool is recommended.
        self._pool: asyncpg.Pool | None = None

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

    async def create_new_pool(
            self,
            database_name: Optional[str] = None, 
            min_size: int = 10,
            max_size: int = 20) -> asyncpg.Pool:
        """Create a new connection pool for the specified database."""
        if database_name is None and self._database_name is None:
            raise ValueError(
                "No database name provided and no default database name set")
        elif database_name is None:
            database_name = self._database_name
        else:
            self._database_name = database_name
      
        await self.close_pool()
        
        self._pool = await asyncpg.create_pool(
            f"{self._server_data_source_name}/{database_name}",
            min_size=min_size,
            max_size=max_size
        )
        return self._pool

    async def close_pool(self):
        """Close the connection pool if it exists."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    @asynccontextmanager
    async def connect(self, database_name: Optional[str] = None) -> \
        AsyncGenerator[Any, None]:
        """Connect using pool if available, otherwise create individual connection."""
        if database_name is None and self._database_name is None:
            raise ValueError(
                "No database name provided and no default database name set")
        elif database_name is None:
            database_name = self._database_name
        else:
            self._database_name = database_name

        database_exists = await self.database_exists(database_name)

        if not database_exists:
            raise ValueError(f"Database {database_name} does not exist")

        # Use pool if available, otherwise create individual connection
        if self._pool is not None and self._database_name == database_name:
            # Use pool
            async with self._pool.acquire() as conn:
                yield conn
        else:
            # Fall back to individual connection (existing behavior)
            if self._connection is not None and not self._connection.is_closed():
                await self._connection.close()
                self._connection = None

            self._connection = await asyncpg.connect(
                f"{self._server_data_source_name}/{database_name}")
            try:
                yield self._connection
            # Statements inside finally are called once the context manager
            # exists, e.g.
            # async with postgres.connect() as conn:
            #     await conn.execute("SELECT 1")
            finally:
                if self._connection is not None:
                    await self._connection.close()
                    self._connection = None

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
        """Close both individual connection and pool."""
        await self.close_pool()
        if self._connection is not None:
            if not self._connection.is_closed():
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

    async def create_table_from_schema(self, schema: str):
        """Create a table from a schema string."""
        async with self._pool.acquire() as conn:
            await conn.execute(schema)

    async def list_tables(self, database_name: Optional[str] = None) \
        -> List[Dict[str, str]]:
        """
        List all tables in the database.
        
        Args:
            database_name: Optional database name to connect to
            
        Returns:
            List of dictionaries containing table information
            
        Raises:
            ValueError: If no connection is available and database doesn't exist
        """
        # Determine which connection to use
        conn = await self._get_connection_for_operation(database_name)
        
        # Query to get table information
        query = """
            SELECT 
                schemaname,
                tablename,
                tableowner,
                tablespace,
                hasindexes,
                hasrules,
                hastriggers,
                rowsecurity
            FROM pg_tables 
            WHERE schemaname NOT IN ('information_schema', 'pg_catalog')
            ORDER BY schemaname, tablename;
        """
        
        rows = await conn.fetch(query)
        
        # Convert to list of dictionaries
        tables = []
        for row in rows:
            tables.append({
                'schema': row['schemaname'],
                'name': row['tablename'],
                'owner': row['tableowner'],
                'tablespace': row['tablespace'],
                'has_indexes': row['hasindexes'],
                'has_rules': row['hasrules'],
                'has_triggers': row['hastriggers'],
                'row_security': row['rowsecurity']
            })
        
        return tables

    async def _get_connection_for_operation(
            self,
            database_name: Optional[str] = None) -> asyncpg.Connection:
        """Get the appropriate connection for database operations."""
        target_db = database_name or self._database_name
        
        if target_db is None:
            raise ValueError(
                "No database name specified and no default database set")
        
        if not await self.database_exists(target_db):
            raise ValueError(f"Database '{target_db}' does not exist")
        
        # Use existing individual connection if available
        if self._connection is not None and not self._connection.is_closed():
            return self._connection
        
        # Use pool if available and database matches
        if self._pool is not None and self._database_name == target_db:
            # Acquire connection from pool and return it
            return await self._pool.acquire()
        
        # Create temporary connection
        return await asyncpg.connect(
            f"{self._server_data_source_name}/{target_db}")

    async def list_table_names(self, database_name: Optional[str] = None) \
        -> List[str]:
        """
        List just the table names in the database.
        
        Args:
            database_name: Optional database name to connect to
            
        Returns:
            List of table names
        """
        conn = await self._get_connection_for_operation(database_name)
        
        query = """
            SELECT tablename 
            FROM pg_tables 
            WHERE schemaname NOT IN ('information_schema', 'pg_catalog')
            ORDER BY tablename;
        """
        
        rows = await conn.fetch(query)
        return [row['tablename'] for row in rows]

    async def is_table_exists(
            self,
            table_name: str,
            database_name: Optional[str] = None,
            schema_name: str = 'public',
            ) -> bool:
        """Check if a table exists in the current database."""
        conn = await self._get_connection_for_operation(database_name)
        
        exists = await conn.fetchval("""
            SELECT 1 FROM information_schema.tables 
            WHERE table_schema = $1 AND table_name = $2
        """, schema_name, table_name)
        
        return exists is not None
