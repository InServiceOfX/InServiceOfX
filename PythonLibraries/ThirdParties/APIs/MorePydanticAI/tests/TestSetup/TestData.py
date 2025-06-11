def pydantic_ai_sql_generation_test_data():

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