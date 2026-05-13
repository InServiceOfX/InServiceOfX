CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY,
    source_path TEXT NOT NULL,
    source_sha256 TEXT,
    title TEXT,
    metadata JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(source_path, source_sha256)
);

CREATE TABLE IF NOT EXISTS document_pages (
    id UUID PRIMARY KEY,
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page_number INTEGER NOT NULL,
    image_path TEXT,
    width INTEGER,
    height INTEGER,
    metadata JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(document_id, page_number)
);

CREATE TABLE IF NOT EXISTS page_embedding_blobs (
    id UUID PRIMARY KEY,
    page_id UUID NOT NULL REFERENCES document_pages(id) ON DELETE CASCADE,
    model_name TEXT NOT NULL,
    model_revision TEXT,
    embedding_shape INTEGER[] NOT NULL,
    storage_format TEXT NOT NULL DEFAULT 'safetensors',
    embedding_bytes BYTEA NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(page_id, model_name, model_revision)
);

CREATE INDEX IF NOT EXISTS idx_document_pages_document_id
    ON document_pages(document_id);

CREATE INDEX IF NOT EXISTS idx_page_embedding_blobs_page_id
    ON page_embedding_blobs(page_id);

CREATE INDEX IF NOT EXISTS idx_page_embedding_blobs_model_name
    ON page_embedding_blobs(model_name);
