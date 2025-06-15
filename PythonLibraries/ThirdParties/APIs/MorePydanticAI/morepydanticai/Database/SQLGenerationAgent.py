"""
https://ai.pydantic.dev/examples/sql-gen/#example-code
"""

from annotated_types import MinLen
from dataclasses import dataclass
from datetime import date
from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry, RunContext, format_as_xml
from typing import Annotated, Union, List, Dict
from typing_extensions import TypeAlias

import asyncpg

@dataclass
class Dependencies:
    connection: asyncpg.Connection

# Use pydantic and BaseModel for
# * BaseModel validates data at runtime
# * automatic type conversion of data
# * Pydantic models can automatically generate JSON schemas
# * Pydantic's Field provides metadata and validation
# * Serialization/Deserialization, to and from JSON, dicts
#   - dataclasses require manual serialization code.

class Success(BaseModel):
    """Response when SQL could be successfully generated."""

    sql_query: Annotated[str, MinLen(1)]
    explanation: str = Field(
        '', description='Explanation of the SQL query, as markdown'
    )

class InvalidRequest(BaseModel):
    """Response when the user input didn't include enough information to 
    generate SQL."""

    error: str

Response: TypeAlias = Union[Success, InvalidRequest]

def create_agent(
        db_schema: str,
        sql_examples: List[Dict[str, str]],
        model: str = None):
    if model is None:
        model = 'groq:gemma2-9b-it'
    agent = Agent[Dependencies, Response](
        model,
        # Type ignore while we wait for PEP-0747, nonetheless unions will work
        # fine everywhere else
        response_format=Response,
        deps_type=Dependencies,
    )

    async def system_prompt() -> str:
        return f"""\
You are a SQL query generator. Your task is to generate valid PostgreSQL SQL queries based on user requests.

IMPORTANT: You must return a JSON object with one of these two structures:

For valid SQL queries:
{{
    "sql_query": "your SQL query here",
    "explanation": "optional explanation here"
}}

For invalid requests (when the request cannot be converted to a valid SQL query):
{{
    "error": "clear explanation of why the request cannot be converted to SQL"
}}

Examples of invalid requests:
- Requests for non-existent tables or columns
- Requests that are too vague to generate SQL
- Requests that are not related to database queries
- Requests for operations not supported by the schema

Database schema:

{db_schema}

Today's date = {date.today()}

Example queries and their corresponding SQL:
{format_as_xml(sql_examples)}

Remember:
1. Return a JSON object with either sql_query/explanation or error fields
2. For invalid requests, use the error format
3. The sql_query must be a SELECT statement
4. The sql_query must be valid PostgreSQL syntax
5. Use the exact table and column names from the schema
6. For JSONB fields, use the -> operator for JSON objects and ->> for text values
"""
    async def validate_output(
            context: RunContext[Dependencies],
            output: Response) -> Response:
        if isinstance(output, InvalidRequest):
            return output

        # TODO: gemini often adds extraneous backslashes to SQL. Try it
        # empirically.
        output.sql_query = output.sql_query.replace('\\', '')
        if not output.sql_query.upper().startswith('SELECT'):
            raise ModelRetry('Please create a SELECT query')

        try:
            # execute is a method that sends SQL commands to the PostgreSQL
            # server.
            await context.deps.connection.execute(
            # EXPLAIN is a PostgreSQL command that shows the execution plan,
            # "dry run", without actually running it.
                f'EXPLAIN {output.sql_query}')
        except asyncpg.exceptions.PostgresError as e:
            raise ModelRetry(f'Invalid query: {e}') from e
        else:
            return output

    # Register the functions with the agent. This replaces run time decorators
    # @agent.system_prompt
    # async def system_prompt() -> str:
    agent.system_prompt = system_prompt

    agent.output_validator = validate_output

    return agent