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

    VECTOR_SIMILARITY_SEARCH_MESSAGE_CHUNKS = """
    SELECT id, conversation_id, chunk_index, total_chunks, parent_message_hash, 
           content, datetime, hash, role, embedding, created_at,
           1 - (embedding <=> $1) as similarity_score
    FROM permanent_conversation_message_chunks 
    WHERE ($2::text IS NULL OR role = $2::text)  -- Explicit text casting
    AND ($3::float IS NULL OR (1 - (embedding <=> $1)) >= $3::float) -- to float
    ORDER BY embedding <=> $1
    LIMIT $4;
    """

    # Vector similarity search with datetime filter
    VECTOR_SIMILARITY_SEARCH_MESSAGE_CHUNKS_WITH_DATETIME_FILTER = """
    SELECT id, conversation_id, chunk_index, total_chunks, parent_message_hash, 
        content, datetime, hash, role, embedding, created_at,
        1 - (embedding <=> $1) as similarity_score
    FROM permanent_conversation_message_chunks 
    WHERE ($2::float IS NULL OR datetime >= $2::float)
    AND ($3::float IS NULL OR datetime <= $3::float)
    AND ($4::text IS NULL OR role = $4::text)
    AND ($5::float IS NULL OR (1 - (embedding <=> $1)) >= $5::float)
    ORDER BY embedding <=> $1
    LIMIT $6;
    """

    # Combined vector similarity search
    VECTOR_SIMILARITY_SEARCH_COMBINED = """
    WITH message_chunks AS (
        SELECT id, conversation_id, chunk_index, total_chunks, parent_message_hash, 
            content, datetime, hash, role, embedding, created_at,
            1 - (embedding <=> $1) as similarity_score,
            'message' as chunk_type
        FROM permanent_conversation_message_chunks 
        WHERE ($2 IS NULL OR datetime >= $2)  -- Optional start_date filter
        AND ($3 IS NULL OR datetime <= $3)  -- Optional end_date filter
        AND ($4 IS NULL OR role = $4)  -- Optional role filter
        AND ($5 IS NULL OR (1 - (embedding <=> $1)) >= $5)  -- Optional similarity threshold
    ),
    message_pair_chunks AS (
        SELECT id, conversation_id, chunk_index, total_chunks, parent_message_hash, 
            content, datetime, hash, role, embedding, created_at,
            1 - (embedding <=> $1) as similarity_score,
            'message_pair' as chunk_type
        FROM permanent_conversation_message_pair_chunks 
        WHERE ($2 IS NULL OR datetime >= $2)  -- Optional start_date filter
        AND ($3 IS NULL OR datetime <= $3)  -- Optional end_date filter
        AND ($4 IS NULL OR role = $4)  -- Optional role filter
        AND ($5 IS NULL OR (1 - (embedding <=> $1)) >= $5)  -- Optional similarity threshold
    ),
    combined_results AS (
        SELECT * FROM message_chunks
        UNION ALL
        SELECT * FROM message_pair_chunks
    )
    SELECT * FROM combined_results
    ORDER BY similarity_score DESC
    LIMIT $6;
    """

    # Get all message chunks from the table
    GET_ALL_MESSAGE_CHUNKS = """
    SELECT id, conversation_id, chunk_index, total_chunks, parent_message_hash, 
           content, datetime, hash, role, embedding, created_at
    FROM permanent_conversation_message_chunks 
    ORDER BY conversation_id, chunk_index;
    """

    # Get all message chunks from the table
    GET_ALL_MESSAGE_PAIR_CHUNKS = """
    SELECT id, conversation_id, chunk_index, total_chunks, parent_message_hash, 
           content, datetime, hash, role, embedding, created_at
    FROM permanent_conversation_message_pair_chunks 
    ORDER BY conversation_id, chunk_index;
    """

    # Get latest N message chunks ordered by datetime
    GET_LATEST_MESSAGE_CHUNKS = """
    SELECT id, conversation_id, chunk_index, total_chunks, parent_message_hash,
        content, datetime, hash, role, embedding, created_at
    FROM permanent_conversation_message_chunks
    ORDER BY datetime DESC
    LIMIT $1;
    """

    # Get latest N message pair chunks ordered by datetime
    GET_LATEST_MESSAGE_PAIR_CHUNKS = """
    SELECT id, conversation_id, chunk_index, total_chunks, parent_message_hash,
        content, datetime, hash, role, embedding, created_at
    FROM permanent_conversation_message_pair_chunks
    ORDER BY datetime DESC
    LIMIT $1;
    """
