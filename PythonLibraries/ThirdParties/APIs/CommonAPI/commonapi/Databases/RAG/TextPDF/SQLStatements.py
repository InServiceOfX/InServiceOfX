class SQLStatements:
    CREATE_TEXT_PDFS_TABLE = """
    CREATE TABLE IF NOT EXISTS text_pdfs (
        id SERIAL PRIMARY KEY,
        filename VARCHAR(255) NOT NULL UNIQUE,
        total_chunks INTEGER NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """

    CREATE_TEXT_PDF_CHUNKS_TABLE = """
    CREATE TABLE IF NOT EXISTS text_pdf_chunks (
        id SERIAL PRIMARY KEY,
        document_id INTEGER REFERENCES text_pdfs(id),
        chunk_id INTEGER NOT NULL,
        content TEXT NOT NULL,
        page_number INTEGER,
        chunk_hash VARCHAR(64) UNIQUE NOT NULL,
        embedding VECTOR(1024),
        metadata JSONB,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """

    CREATE_INDEX_FOR_SIMILARITY_SEARCH = """
    CREATE INDEX IF NOT EXISTS idx_text_pdf_chunks_embedding 
        ON text_pdf_chunks USING hnsw (embedding vector_l2_ops);
    """

    CREATE_INDEX_FOR_TEXT_PDF_CHUNKS_CONTENT = """
    CREATE INDEX IF NOT EXISTS idx_text_pdf_chunks_content 
        ON text_pdf_chunks USING gin (content);
    """

    INSERT_TEXT_PDF = """
    INSERT INTO text_pdfs (filename, total_chunks)
    VALUES ($1, $2)
    ON CONFLICT (filename) DO UPDATE SET
        total_chunks = EXCLUDED.total_chunks
    RETURNING id;
    """

    INSERT_TEXT_PDF_CHUNK = """
    INSERT INTO text_pdf_chunks (document_id, chunk_id, content, page_number, chunk_hash, embedding, metadata)
    VALUES ($1, $2, $3, $4, $5, $6, $7)
    ON CONFLICT (chunk_hash) DO NOTHING
    RETURNING id;
    """

    SELECT_TEXT_PDF_CHUNKS_BY_EMBEDDING = """
    SELECT 
        c.id,
        c.content,
        c.page_number,
        c.metadata,
        d.filename,
        c.embedding <=> $1 as distance
    FROM text_pdf_chunks c
    JOIN text_pdfs d ON c.document_id = d.id
    ORDER BY c.embedding <=> $1
    LIMIT $2;
    """

    SELECT_TEXT_PDF_CHUNKS_BY_FILENAME = """
    SELECT 
        c.chunk_id,
        c.content,
        c.page_number,
        c.metadata
    FROM text_pdf_chunks c
    JOIN text_pdfs d ON c.document_id = d.id
    WHERE d.filename = $1
    ORDER BY c.chunk_id;
    """
                
