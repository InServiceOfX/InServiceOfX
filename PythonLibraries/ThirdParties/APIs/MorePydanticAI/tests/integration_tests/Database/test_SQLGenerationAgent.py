from corecode.Utilities import load_environment_file
import pytest
import pytest_asyncio
import asyncpg
from morepydanticai.Database import PostgreSQLConnection, SQLGenerationAgent
from TestSetup.TestData import pydantic_ai_sql_generation_test_data
from datetime import datetime
# for timezone handling
import pytz
import json

load_environment_file()

# Reuse your existing connection parameters
DATABASE_PORT = 5432
IP_ADDRESS = "192.168.86.201"
POSTGRES_USER = "inserviceofx"
POSTGRES_PASSWORD = "mypassword"
TEST_DSN = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{IP_ADDRESS}:{DATABASE_PORT}"
TEST_DB_NAME = "test_sql_generation"

EXAMPLE_RECORDS = [
    {
        "created_at": datetime(2024, 3, 20, 10, 0, 0, tzinfo=pytz.UTC),
        "start_timestamp": datetime(2024, 3, 20, 10, 0, 0, tzinfo=pytz.UTC),
        "end_timestamp": datetime(2024, 3, 20, 10, 10, 0, tzinfo=pytz.UTC),
        "trace_id": "trace1",
        "span_id": "span1",
        "parent_span_id": None,
        "level": "error",
        "span_name": "test_span",
        "message": "Test error message",
        "attributes_json_schema": '{"type": "object"}',
        "attributes": json.dumps({"foobar": False, "test_key": "test_value"}),
        "tags": ["test", "error"],
        "is_exception": True,
        "otel_status_message": "Error occurred",
        "service_name": "test_service"
    },
    {
        "created_at": datetime(2024, 3, 20, 11, 0, 0, tzinfo=pytz.UTC),
        "start_timestamp": datetime(2024, 3, 20, 11, 0, 0, tzinfo=pytz.UTC),
        "end_timestamp": datetime(2024, 3, 20, 11, 10, 0, tzinfo=pytz.UTC),
        "trace_id": "trace2",
        "span_id": "span2",
        "parent_span_id": None,
        "level": "info",
        "span_name": "test_span2",
        "message": "Test info message",
        "attributes_json_schema": '{"type": "object"}',
        "attributes": json.dumps({"foobar": True, "test_key": "test_value2"}),
        "tags": ["test", "info"],
        "is_exception": False,
        "otel_status_message": "Success",
        "service_name": "test_service"
    }
]

@pytest_asyncio.fixture(scope="function")
async def postgres_connection():
    """Create a PostgreSQLConnection instance for testing."""
    conn = PostgreSQLConnection(TEST_DSN, TEST_DB_NAME)
    
    # Create the database and schema
    await conn.create_database(TEST_DB_NAME)
    async with conn.connect() as pg_conn:
        # Create the log_level type if it doesn't exist
        await pg_conn.execute("""
            DO $$ BEGIN
                CREATE TYPE log_level AS ENUM ('debug', 'info', 'warning', 'error', 'critical');
            EXCEPTION
                WHEN duplicate_object THEN null;
            END $$;
        """)
        
        # Create the table
        await pg_conn.execute("""
            CREATE TABLE IF NOT EXISTS records (
                created_at timestamptz,
                start_timestamp timestamptz,
                end_timestamp timestamptz,
                trace_id text,
                span_id text,
                parent_span_id text,
                level log_level,
                span_name text,
                message text,
                attributes_json_schema text,
                attributes jsonb,
                tags text[],
                is_exception boolean,
                otel_status_message text,
                service_name text
            );
        """)
        
        # Insert example data
        for record in EXAMPLE_RECORDS:
            values = list(record.values())
            await pg_conn.execute("""
                INSERT INTO records (
                    created_at, start_timestamp, end_timestamp, trace_id,
                    span_id, parent_span_id, level, span_name, message,
                    attributes_json_schema, attributes, tags, is_exception,
                    otel_status_message, service_name
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15
                )
            """, *values)
    
    yield conn
    
    await cleanup_test_database(TEST_DB_NAME)

async def cleanup_test_database(db_name: str):
    """Clean up test database by dropping it and terminating all connections."""
    sys_conn = await asyncpg.connect(
        TEST_DSN + "/" + PostgreSQLConnection.DEFAULT_SYSTEM_DB)
    try:
        await sys_conn.execute(f"""
            SELECT pg_terminate_backend(pid) 
            FROM pg_stat_activity 
            WHERE datname = $1
        """, db_name)
        await sys_conn.execute(f'DROP DATABASE IF EXISTS {db_name}')
    finally:
        await sys_conn.close()

@pytest.mark.asyncio
async def test_sql_generation_basic(postgres_connection):
    """Test basic SQL generation with a simple query."""
    db_schema, sql_examples = pydantic_ai_sql_generation_test_data()
    
    agent = SQLGenerationAgent.create_agent(
        db_schema=db_schema,
        sql_examples=sql_examples,
        model='groq:gemma2-9b-it',
    )
    
    # Test a simple query with more specific prompt
    async with postgres_connection.connect() as conn:
        deps = SQLGenerationAgent.Dependencies(connection=conn)
        result = await agent.run(
            "Generate a SQL query to select all records where level is 'error'"
        )

        print(result)
        print(type(result))

        assert isinstance(result.output, SQLGenerationAgent.Success)
        assert "SELECT" in result.output.sql_query.upper()
        assert "level" in result.output.sql_query.lower()
        assert "error" in result.output.sql_query.lower()
        
        # Verify the query works
        rows = await conn.fetch(result.output.sql_query)
        assert len(rows) > 0
        assert all(row['level'] == 'error' for row in rows)

@pytest.mark.asyncio
async def test_sql_generation_with_attributes(postgres_connection):
    """Test SQL generation with attribute filtering."""
    db_schema, sql_examples = pydantic_ai_sql_generation_test_data()
    
    agent = SQLGenerationAgent.create_agent(
        db_schema=db_schema,
        sql_examples=sql_examples,
        model='groq:llama-3.1-8b-instant',
    )
    
    async with postgres_connection.connect() as conn:
        deps = SQLGenerationAgent.Dependencies(connection=conn)
        result = await agent.run(
            "Generate a SQL query to select records where the 'foobar' attribute is false"
        )
        print(result)
        print(type(result))

        assert isinstance(result.output, SQLGenerationAgent.Success)
        assert "SELECT" in result.output.sql_query.upper()
        assert "attributes" in result.output.sql_query.lower()
        assert "foobar" in result.output.sql_query.lower()
        
        # Verify the query works
        rows = await conn.fetch(result.output.sql_query)
        assert len(rows) > 0
        assert all(row['attributes']['foobar'] is False for row in rows)

@pytest.mark.asyncio
async def test_sql_generation_invalid_request(postgres_connection):
    """Test SQL generation with an invalid request."""
    db_schema, sql_examples = pydantic_ai_sql_generation_test_data()
    
    agent = SQLGenerationAgent.create_agent(
        db_schema=db_schema,
        sql_examples=sql_examples,
        model='groq:gemma2-9b-it',
    )
    
    async with postgres_connection.connect() as conn:
        deps = SQLGenerationAgent.Dependencies(connection=conn)
        # Make the invalid request more explicit about database context
        result = await agent.run(
            "show me records from a table that doesn't exist in the database"
        )
        
        print(result)
        print(type(result))
        
        assert isinstance(result.output, SQLGenerationAgent.InvalidRequest)
        assert len(result.output.error) > 0
        # Verify the error message is relevant to the database context
        assert "table" in result.output.error.lower()

@pytest.mark.asyncio
async def test_sql_generation_with_date_filter(postgres_connection):
    """Test SQL generation with date filtering."""
    db_schema, sql_examples = pydantic_ai_sql_generation_test_data()
    
    agent = SQLGenerationAgent.create_agent(
        db_schema=db_schema,
        sql_examples=sql_examples,
        model='groq:llama-3.1-8b-instant',
    )
    
    async with postgres_connection.connect() as conn:
        deps = SQLGenerationAgent.Dependencies(connection=conn)
        result = await agent.run("show me records from yesterday")
        
        assert isinstance(result.output, SQLGenerationAgent.Success)
        assert "CURRENT_TIMESTAMP" in result.output.sql_query
        assert "INTERVAL" in result.output.sql_query
        
        # Verify the query works
        rows = await conn.fetch(result.output.sql_query)
        # Note: This might return 0 rows depending on when the test is run
        # since we're using fixed timestamps in our test data
