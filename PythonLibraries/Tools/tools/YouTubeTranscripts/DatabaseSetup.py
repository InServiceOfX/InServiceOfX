class DatabaseSetup:
    CHUNKS_DATABASE_NAME = "youtube_podcast_transcripts_chunks"

    # Enforce no duplicate chunk index for same video.
    #     UNIQUE(video_id, chunk_index)
    CHUNKS_DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    id SERIAL PRIMARY KEY,
    video_id VARCHAR(255) NOT NULL,
    chunk_index INTEGER NOT NULL CHECK (chunk_index >= 0),
    total_number_of_chunks INTEGER NOT NULL CHECK (total_number_of_chunks > 0),
    embedding vector(1024) NOT NULL,
    UNIQUE(video_id, chunk_index),
    CONSTRAINT valid_chunk_index CHECK (chunk_index < total_number_of_chunks),
    CONSTRAINT valid_video_id_length CHECK (LENGTH(video_id) > 0)
);
    """

    def __init__(self, database_name: str):
        self.database_name = database_name

