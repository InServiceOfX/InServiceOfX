"""
https://ai.pydantic.dev/examples/sql-gen/#example-code
"""

from annotated_types import MinLen
from dataclasses import dataclass
from pydantic import BaseModel, Field
from typing import Annotated

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

