class CommonSQLStatements:
    """Static class containing common SQL statements."""
    
    CHECK_TABLE_EXISTS = """
        SELECT 1 FROM information_schema.tables
        WHERE table_schema = 'public' AND table_name = $1;
    """
