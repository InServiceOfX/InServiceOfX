class SQLStatements:
    """Static class containing SQL statements for permanent conversation
    operations."""

    # Messages table
    CREATE_PERMANENT_CONVERSATION_MESSAGES_TABLE = """
    CREATE TABLE IF NOT EXISTS permanent_conversation_messages (
        id SERIAL PRIMARY KEY,
        conversation_id INTEGER NOT NULL,
        content TEXT,
        datetime DOUBLE PRECISION NOT NULL,
        hash VARCHAR(64) NOT NULL,
        role VARCHAR(50) NOT NULL,
        -- bge-large-en-v1.5 returns a vector of 1024 floats
        embedding VECTOR(1024),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """

    # Message pairs table
    CREATE_PERMANENT_CONVERSATION_MESSAGE_PAIRS_TABLE = """
    CREATE TABLE IF NOT EXISTS permanent_conversation_message_pairs (
        id SERIAL PRIMARY KEY,
        conversation_pair_id INTEGER NOT NULL,
        content_0 TEXT,
        content_1 TEXT,
        datetime DOUBLE PRECISION NOT NULL,
        hash VARCHAR(64) NOT NULL,
        role_0 VARCHAR(50) NOT NULL,
        role_1 VARCHAR(50) NOT NULL,
        -- bge-large-en-v1.5 returns a vector of 1024 floats
        embedding VECTOR(1024),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """

    # Insert statements for messages
    INSERT_PERMANENT_CONVERSATION_MESSAGE = """
    INSERT INTO permanent_conversation_messages 
    (conversation_id, content, datetime, hash, role, embedding)
    VALUES ($1, $2, $3, $4, $5, $6)
    RETURNING id;
    """

    # Insert statements for message pairs
    INSERT_PERMANENT_CONVERSATION_MESSAGE_PAIR = """
    INSERT INTO permanent_conversation_message_pairs 
    (conversation_pair_id, content_0, content_1, datetime, hash, role_0, role_1, embedding)
    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
    RETURNING id;
    """

    # Select statements for messages
    SELECT_ALL_PERMANENT_CONVERSATION_MESSAGES = """
    SELECT id, conversation_id, content, datetime, hash, role, embedding
    FROM permanent_conversation_messages
    ORDER BY conversation_id ASC;
    """

    SELECT_PERMANENT_CONVERSATION_MESSAGE_BY_HASH = """
    SELECT id, conversation_id, content, datetime, hash, role, embedding
    FROM permanent_conversation_messages
    WHERE hash = $1;
    """

    # Select statements for message pairs
    SELECT_ALL_PERMANENT_CONVERSATION_MESSAGE_PAIRS = """
    SELECT id, conversation_pair_id, content_0, content_1, datetime, hash, role_0, role_1, embedding
    FROM permanent_conversation_message_pairs
    ORDER BY conversation_pair_id ASC;
    """

    SELECT_PERMANENT_CONVERSATION_MESSAGE_PAIR_BY_HASH = """
    SELECT id, conversation_pair_id, content_0, content_1, datetime, hash, role_0, role_1, embedding
    FROM permanent_conversation_message_pairs
    WHERE hash = $1;
    """

    # Delete statements
    DELETE_PERMANENT_CONVERSATION_MESSAGE_BY_HASH = """
    DELETE FROM permanent_conversation_messages
    WHERE hash = $1
    RETURNING id;
    """

    DELETE_PERMANENT_CONVERSATION_MESSAGE_PAIR_BY_HASH = """
    DELETE FROM permanent_conversation_message_pairs
    WHERE hash = $1
    RETURNING id;
    """

    # Drop statements
    DROP_PERMANENT_CONVERSATION_MESSAGES_TABLE = """
    DROP TABLE IF EXISTS permanent_conversation_messages CASCADE;
    """

    DROP_PERMANENT_CONVERSATION_MESSAGE_PAIRS_TABLE = """
    DROP TABLE IF EXISTS permanent_conversation_message_pairs CASCADE;
    """

    # Utility statements
    GET_MAX_CONVERSATION_ID = """
    SELECT COALESCE(MAX(conversation_id), -1) FROM permanent_conversation_messages;
    """

    GET_MAX_CONVERSATION_PAIR_ID = """
    SELECT COALESCE(MAX(conversation_pair_id), -1) FROM permanent_conversation_message_pairs;
    """