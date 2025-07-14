class SQLStatements:
    """Static class containing SQL statements for permanent conversation
    operations."""

    CREATE_PERMANENT_CONVERSATION_TABLE = """
    CREATE TABLE IF NOT EXISTS permanent_conversation (
        id SERIAL PRIMARY KEY,
        conversation_id UUID NOT NULL,
        timestamp DOUBLE PRECISION NOT NULL,
        role TEXT NOT NULL,
        hash TEXT,
        content TEXT,
        -- bge-large-en-v1.5 returns a vector of 1024 floats
        embedding vector(1024) NOT NULL
    );
    """

    INSERT_PERMANENT_CONVERSATION_MESSAGE = """
    INSERT INTO permanent_conversation (conversation_id, timestamp, role, hash, content, embedding)
    VALUES ($1, $2, $3, $4, $5, $6)
    ON CONFLICT (hash) DO NOTHING
    RETURNING id;
    """

    SELECT_ALL_PERMANENT_CONVERSATION_MESSAGES = """
    SELECT id, conversation_id, timestamp, role, hash, content, embedding
    FROM permanent_conversation
    ORDER BY timestamp ASC;
    """

    SELECT_PERMANENT_CONVERSATION_MESSAGE_BY_HASH = """
    SELECT id, conversation_id, timestamp, role, hash, content, embedding
    FROM permanent_conversation
    WHERE hash = $1;
    """

    DROP_PERMANENT_CONVERSATION_TABLE = """
    DROP TABLE IF EXISTS permanent_conversation CASCADE;
    """