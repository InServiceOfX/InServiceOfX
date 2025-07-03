"""
SQL statements for system messages database operations.
This module contains SQL queries as string statements for system messages table
operations.
"""

class SQLStatements:
    """Static class containing SQL statements for system messages operations."""
    
    # System Messages Table Schema
    CREATE_SYSTEM_MESSAGES_TABLE = """
    CREATE TABLE IF NOT EXISTS system_messages (
        id SERIAL PRIMARY KEY,
        content TEXT NOT NULL,
        timestamp DOUBLE PRECISION NOT NULL,
        hash VARCHAR(64) UNIQUE NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
    
    # System Messages CRUD Operations
    INSERT_SYSTEM_MESSAGE = """
    INSERT INTO system_messages (content, timestamp, hash)
    VALUES ($1, $2, $3)
    ON CONFLICT (hash) DO NOTHING
    RETURNING id;
    """
    
    SELECT_ALL_SYSTEM_MESSAGES = """
    SELECT id, content, timestamp, hash, created_at
    FROM system_messages
    ORDER BY created_at ASC;
    """
    
    SELECT_SYSTEM_MESSAGE_BY_HASH = """
    SELECT id, content, timestamp, hash, created_at
    FROM system_messages
    WHERE hash = $1;
    """
    
    DELETE_SYSTEM_MESSAGE_BY_HASH = """
    DELETE FROM system_messages
    WHERE hash = $1
    RETURNING id;
    """
    
    # Table Management
    DROP_SYSTEM_MESSAGES_TABLE = """
    DROP TABLE IF EXISTS system_messages CASCADE;
    """