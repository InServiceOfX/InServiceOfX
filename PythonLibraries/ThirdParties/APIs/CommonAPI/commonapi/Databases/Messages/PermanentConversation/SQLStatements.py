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
        embedding VECTOR(1536)
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