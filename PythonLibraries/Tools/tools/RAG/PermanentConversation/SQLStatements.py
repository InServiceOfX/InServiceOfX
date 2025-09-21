class SQLStatements:
    """Static class containing SQL statements for permanent conversation
    chunk operations."""

    # Chunks table (replaces individual messages and pairs)
    CREATE_PERMANENT_CONVERSATION_MESSAGE_CHUNKS_TABLE = """
    CREATE TABLE IF NOT EXISTS permanent_conversation_message_chunks (
        id SERIAL PRIMARY KEY,
        conversation_id INTEGER NOT NULL,
        chunk_index INTEGER NOT NULL, -- order within the conversation
        total_chunks INTEGER NOT NULL, -- total number of chunks
        parent_message_hash VARCHAR(64) NOT NULL, -- hash of the parent message
        content TEXT,
        datetime DOUBLE PRECISION, -- timestamp of original message creation
        hash VARCHAR(64) NOT NULL UNIQUE, -- SHA-256 hash
        role VARCHAR(50) NOT NULL, -- 'user', 'assistant', 'system', ...
        embedding VECTOR(1024), -- BGE-large embedding vector
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """

    # Chunks table (replaces individual messages and pairs)
    CREATE_PERMANENT_CONVERSATION_MESSAGE_PAIR_CHUNKS_TABLE = """
    CREATE TABLE IF NOT EXISTS permanent_conversation_message_pair_chunks (
        id SERIAL PRIMARY KEY,
        conversation_id INTEGER NOT NULL,
        chunk_index INTEGER NOT NULL, -- order within the conversation
        total_chunks INTEGER NOT NULL, -- total number of chunks
        parent_message_hash VARCHAR(64) NOT NULL, -- hash of the parent message
        content TEXT,
        datetime DOUBLE PRECISION, -- timestamp of original message pair
        hash VARCHAR(64) NOT NULL UNIQUE, -- SHA-256 hash
        role VARCHAR(50) NOT NULL, -- 'user', 'assistant', 'system', ...
        embedding VECTOR(1024), -- BGE-large embedding vector
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """

    # Create indexes for message chunks table
    CREATE_MESSAGE_CHUNKS_INDEXES = """
    -- HNSW index for vector similarity search (most important!)
    CREATE INDEX IF NOT EXISTS idx_message_chunks_embedding_hnsw 
    ON permanent_conversation_message_chunks
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

    -- B-tree indexes for filtering and sorting
    CREATE INDEX IF NOT EXISTS idx_message_chunks_conversation_id 
    ON permanent_conversation_message_chunks(conversation_id);

    -- KEEP: parent_message_hash index (useful for finding related chunks)
    CREATE INDEX IF NOT EXISTS idx_message_chunks_parent_hash 
    ON permanent_conversation_message_chunks(parent_message_hash);

    CREATE INDEX IF NOT EXISTS idx_message_chunks_role 
    ON permanent_conversation_message_chunks(role);

    CREATE INDEX IF NOT EXISTS idx_message_chunks_created_at 
    ON permanent_conversation_message_chunks(created_at);

    CREATE INDEX IF NOT EXISTS idx_message_chunks_datetime 
    ON permanent_conversation_message_chunks(datetime);

    -- DON'T CREATE: hash index (PostgreSQL creates it automatically due to UNIQUE constraint)
    """

    # Create indexes for message pair chunks table
    CREATE_MESSAGE_PAIR_CHUNKS_INDEXES = """
    -- HNSW index for vector similarity search (most important!)
    CREATE INDEX IF NOT EXISTS idx_message_pair_chunks_embedding_hnsw 
    ON permanent_conversation_message_pair_chunks 
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

    -- B-tree indexes for filtering and sorting
    CREATE INDEX IF NOT EXISTS idx_message_pair_chunks_conversation_id 
    ON permanent_conversation_message_pair_chunks(conversation_id);

    -- KEEP: parent_message_hash index (useful for finding related chunks)
    CREATE INDEX IF NOT EXISTS idx_message_pair_chunks_parent_hash 
    ON permanent_conversation_message_pair_chunks(parent_message_hash);

    CREATE INDEX IF NOT EXISTS idx_message_pair_chunks_role 
    ON permanent_conversation_message_pair_chunks(role);

    CREATE INDEX IF NOT EXISTS idx_message_pair_chunks_created_at 
    ON permanent_conversation_message_pair_chunks(created_at);

    CREATE INDEX IF NOT EXISTS idx_message_pair_chunks_datetime 
    ON permanent_conversation_message_pair_chunks(datetime);

    -- DON'T CREATE: hash index (PostgreSQL creates it automatically due to UNIQUE constraint)
    """

    # Insert a single message chunk
    INSERT_MESSAGE_CHUNK = """
    INSERT INTO permanent_conversation_message_chunks (
        conversation_id, chunk_index, total_chunks, parent_message_hash, content, 
        datetime, hash, role, embedding
    ) VALUES (
        $1, $2, $3, $4, $5, $6, $7, $8, $9
    ) RETURNING id;
    """

    # Insert a single message pair chunk
    INSERT_MESSAGE_PAIR_CHUNK = """
    INSERT INTO permanent_conversation_message_pair_chunks (
        conversation_id, chunk_index, total_chunks, parent_message_hash, content, 
        datetime, hash, role, embedding
    ) VALUES (
        $1, $2, $3, $4, $5, $6, $7, $8, $9
    ) RETURNING id;
    """

    # Get chunks by conversation ID
    GET_CHUNKS_BY_CONVERSATION = """
    SELECT id, conversation_id, chunk_type, chunk_index, role, 
           content, content_hash, embedding, created_at, updated_at
    FROM permanent_conversation_chunks 
    WHERE conversation_id = $1 
    ORDER BY chunk_index;
    """

    # Get chunks by conversation ID and chunk type
    GET_CHUNKS_BY_CONVERSATION_AND_TYPE = """
    SELECT id, conversation_id, chunk_type, chunk_index, role, 
           content, content_hash, embedding, created_at, updated_at
    FROM permanent_conversation_chunks 
    WHERE conversation_id = $1 AND chunk_type = $2
    ORDER BY chunk_index;
    """

    # Get chunks by conversation ID and role
    GET_CHUNKS_BY_CONVERSATION_AND_ROLE = """
    SELECT id, conversation_id, chunk_type, chunk_index, role, 
           content, content_hash, embedding, created_at, updated_at
    FROM permanent_conversation_chunks 
    WHERE conversation_id = $1 AND role = $2
    ORDER BY chunk_index;
    """

    # Vector similarity search - find most similar chunks
    VECTOR_SIMILARITY_SEARCH = """
    SELECT id, conversation_id, chunk_type, chunk_index, role, 
           content, content_hash, embedding, created_at, updated_at,
           1 - (embedding <=> $1) as similarity_score
    FROM permanent_conversation_chunks 
    WHERE conversation_id = $2
    ORDER BY embedding <=> $1
    LIMIT $3;
    """

    # Vector similarity search across all conversations
    VECTOR_SIMILARITY_SEARCH_ALL = """
    SELECT id, conversation_id, chunk_type, chunk_index, role, 
           content, content_hash, embedding, created_at, updated_at,
           1 - (embedding <=> $1) as similarity_score
    FROM permanent_conversation_chunks 
    ORDER BY embedding <=> $1
    LIMIT $2;
    """

    # Vector similarity search with minimum similarity threshold
    VECTOR_SIMILARITY_SEARCH_WITH_THRESHOLD = """
    SELECT id, conversation_id, chunk_type, chunk_index, role, 
           content, content_hash, embedding, created_at, updated_at,
           1 - (embedding <=> $1) as similarity_score
    FROM permanent_conversation_chunks 
    WHERE conversation_id = $2 AND (1 - (embedding <=> $1)) >= $3
    ORDER BY embedding <=> $1
    LIMIT $4;
    """

    # Get chunk by ID
    GET_CHUNK_BY_ID = """
    SELECT id, conversation_id, chunk_type, chunk_index, role, 
           content, content_hash, embedding, created_at, updated_at
    FROM permanent_conversation_chunks 
    WHERE id = $1;
    """

    # Update chunk content and hash
    UPDATE_CHUNK_CONTENT = """
    UPDATE permanent_conversation_chunks 
    SET content = $2, content_hash = $3, updated_at = CURRENT_TIMESTAMP
    WHERE id = $1;
    """

    # Delete chunks by conversation ID
    DELETE_CHUNKS_BY_CONVERSATION = """
    DELETE FROM permanent_conversation_chunks 
    WHERE conversation_id = $1;
    """

    # Delete chunk by ID
    DELETE_CHUNK_BY_ID = """
    DELETE FROM permanent_conversation_chunks 
    WHERE id = $1;
    """

    # Check if conversation exists
    CONVERSATION_EXISTS = """
    SELECT EXISTS(
        SELECT 1 FROM permanent_conversation_chunks 
        WHERE conversation_id = $1
    );
    """

    # Get all unique conversation IDs
    GET_ALL_CONVERSATION_IDS = """
    SELECT DISTINCT conversation_id 
    FROM permanent_conversation_chunks 
    ORDER BY conversation_id;
    """

    # Get chunks by content hash (since hash is not unique, this returns multiple)
    GET_CHUNKS_BY_HASH = """
    SELECT id, conversation_id, chunk_type, chunk_index, role, 
           content, content_hash, embedding, created_at, updated_at
    FROM permanent_conversation_chunks 
    WHERE content_hash = $1
    ORDER BY created_at;
    """

    # Count chunks by conversation
    COUNT_CHUNKS_BY_CONVERSATION = """
    SELECT COUNT(*) 
    FROM permanent_conversation_chunks 
    WHERE conversation_id = $1;
    """

    # Get recent chunks
    GET_RECENT_CHUNKS = """
    SELECT id, conversation_id, chunk_type, chunk_index, role, 
           content, content_hash, embedding, created_at, updated_at
    FROM permanent_conversation_chunks 
    ORDER BY created_at DESC
    LIMIT $1;
    """

