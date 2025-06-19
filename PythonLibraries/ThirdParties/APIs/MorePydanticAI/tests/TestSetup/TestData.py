def pydantic_ai_sql_generation_test_data():
    """
    Ref:
    https://ai.pydantic.dev/examples/sql-gen/#example-code
    """

    DB_SCHEMA = """
CREATE TABLE records (
    created_at timestamptz,
    start_timestamp timestamptz,
    end_timestamp timestamptz,
    trace_id text,
    span_id text,
    parent_span_id text,
    level log_level,
    span_name text,
    message text,
    attributes_json_schema text,
    attributes jsonb,
    tags text[],
    is_exception boolean,
    otel_status_message text,
    service_name text
);
"""
    SQL_EXAMPLES = [
        {
            'request': 'show me records where foobar is false',
            'response': \
                "SELECT * FROM records WHERE attributes->>'foobar' = false",
        },
        {
            'request': \
                'show me records where attributes include the key "foobar"',
            'response': "SELECT * FROM records WHERE attributes ? 'foobar'",
        },
        {
            'request': 'show me records from yesterday',
            'response': \
                "SELECT * FROM records WHERE start_timestamp::date > CURRENT_TIMESTAMP - INTERVAL '1 day'",
        },
        {
            'request': 'show me error records with the tag "foobar"',
            'response': \
                "SELECT * FROM records WHERE level = 'error' and 'foobar' = ANY(tags)",
        },
    ]

    return DB_SCHEMA, SQL_EXAMPLES

def pydantic_ai_rag_test_data():
    DATABASE_NAME = "pydantic_ai_rag"

    ORIGINAL_DB_SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS doc_sections (
    id serial PRIMARY KEY,
    url text NOT NULL UNIQUE,
    title text NOT NULL,
    content text NOT NULL,
    -- text-embedding-3-small returns a vector of 1536 floats
    embedding vector(1536) NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_doc_sections_embedding ON doc_sections USING hnsw (embedding vector_l2_ops);
"""

    DB_SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS doc_sections (
    id serial PRIMARY KEY,
    url text NOT NULL UNIQUE,
    title text NOT NULL,
    content text NOT NULL,
    -- bge-large-en-v1.5 returns a vector of 1024 floats
    embedding vector(1024) NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_doc_sections_embedding ON doc_sections USING hnsw (embedding vector_l2_ops);
"""

    # JSON document from
    # https://gist.github.com/samuelcolvin/4b5bb9bb163b1122ff17e29e48c10992
    DOCS_JSON = (
        'https://gist.githubusercontent.com/'
        'samuelcolvin/4b5bb9bb163b1122ff17e29e48c10992/raw/'
        '80c5925c42f1442c24963aaf5eb1a324d47afe95/logfire_docs.json')

    return DATABASE_NAME, ORIGINAL_DB_SCHEMA, DB_SCHEMA, DOCS_JSON

