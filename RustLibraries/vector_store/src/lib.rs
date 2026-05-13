use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sqlx::{Error, PgPool};
use std::path::Path;
use uuid::Uuid;

pub const DEFAULT_MODEL_NAME_COLQWEN2_5_V0_2: &str = "vidore/colqwen2.5-v0.2";

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct Document {
    pub id: Uuid,
    pub source_path: String,
    pub source_sha256: Option<String>,
    pub title: Option<String>,
    pub metadata: Value,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct DocumentPage {
    pub id: Uuid,
    pub document_id: Uuid,
    pub page_number: i32,
    pub image_path: Option<String>,
    pub width: Option<i32>,
    pub height: Option<i32>,
    pub metadata: Value,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct PageEmbeddingBlob {
    pub id: Uuid,
    pub page_id: Uuid,
    pub model_name: String,
    pub model_revision: Option<String>,
    pub embedding_shape: Vec<i32>,
    pub storage_format: String,
    pub embedding_bytes: Vec<u8>,
    pub metadata: Value,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone)]
pub struct NewDocument {
    pub source_path: String,
    pub source_sha256: Option<String>,
    pub title: Option<String>,
    pub metadata: Value,
}

#[derive(Debug, Clone)]
pub struct NewDocumentPage {
    pub document_id: Uuid,
    pub page_number: i32,
    pub image_path: Option<String>,
    pub width: Option<i32>,
    pub height: Option<i32>,
    pub metadata: Value,
}

#[derive(Debug, Clone)]
pub struct NewPageEmbeddingBlob {
    pub page_id: Uuid,
    pub model_name: String,
    pub model_revision: Option<String>,
    pub embedding_shape: Vec<i32>,
    pub storage_format: String,
    pub embedding_bytes: Vec<u8>,
    pub metadata: Value,
}

pub async fn create_pool(database_url: &str) -> Result<PgPool, Error> {
    PgPool::connect(database_url).await
}

pub async fn ensure_schema(pool: &PgPool) -> Result<(), Error> {
    for statement in include_str!("../sql/schema.sql")
        .split(';')
        .map(str::trim)
        .filter(|statement| !statement.is_empty())
    {
        sqlx::query(statement).execute(pool).await?;
    }
    Ok(())
}

pub async fn upsert_document(pool: &PgPool, document: NewDocument) -> Result<Document, Error> {
    sqlx::query_as::<_, Document>(
        r#"
        INSERT INTO documents (
            id, source_path, source_sha256, title, metadata
        )
        VALUES ($1, $2, $3, $4, $5)
        ON CONFLICT (source_path, source_sha256) DO UPDATE SET
            title = EXCLUDED.title,
            metadata = EXCLUDED.metadata,
            updated_at = CURRENT_TIMESTAMP
        RETURNING id, source_path, source_sha256, title, metadata,
                  created_at, updated_at
        "#,
    )
    .bind(Uuid::new_v4())
    .bind(document.source_path)
    .bind(document.source_sha256)
    .bind(document.title)
    .bind(document.metadata)
    .fetch_one(pool)
    .await
}

pub async fn upsert_document_page(
    pool: &PgPool,
    page: NewDocumentPage,
) -> Result<DocumentPage, Error> {
    sqlx::query_as::<_, DocumentPage>(
        r#"
        INSERT INTO document_pages (
            id, document_id, page_number, image_path, width, height, metadata
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7)
        ON CONFLICT (document_id, page_number) DO UPDATE SET
            image_path = EXCLUDED.image_path,
            width = EXCLUDED.width,
            height = EXCLUDED.height,
            metadata = EXCLUDED.metadata,
            updated_at = CURRENT_TIMESTAMP
        RETURNING id, document_id, page_number, image_path, width, height,
                  metadata, created_at, updated_at
        "#,
    )
    .bind(Uuid::new_v4())
    .bind(page.document_id)
    .bind(page.page_number)
    .bind(page.image_path)
    .bind(page.width)
    .bind(page.height)
    .bind(page.metadata)
    .fetch_one(pool)
    .await
}

pub async fn upsert_page_embedding_blob(
    pool: &PgPool,
    embedding: NewPageEmbeddingBlob,
) -> Result<PageEmbeddingBlob, Error> {
    sqlx::query_as::<_, PageEmbeddingBlob>(
        r#"
        INSERT INTO page_embedding_blobs (
            id, page_id, model_name, model_revision, embedding_shape,
            storage_format, embedding_bytes, metadata
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        ON CONFLICT (page_id, model_name, model_revision) DO UPDATE SET
            embedding_shape = EXCLUDED.embedding_shape,
            storage_format = EXCLUDED.storage_format,
            embedding_bytes = EXCLUDED.embedding_bytes,
            metadata = EXCLUDED.metadata,
            updated_at = CURRENT_TIMESTAMP
        RETURNING id, page_id, model_name, model_revision, embedding_shape,
                  storage_format, embedding_bytes, metadata,
                  created_at, updated_at
        "#,
    )
    .bind(Uuid::new_v4())
    .bind(embedding.page_id)
    .bind(embedding.model_name)
    .bind(embedding.model_revision)
    .bind(embedding.embedding_shape)
    .bind(embedding.storage_format)
    .bind(embedding.embedding_bytes)
    .bind(embedding.metadata)
    .fetch_one(pool)
    .await
}

pub async fn store_safetensors_embedding_file(
    pool: &PgPool,
    page_id: Uuid,
    model_name: &str,
    model_revision: Option<&str>,
    embedding_shape: Vec<i32>,
    path: impl AsRef<Path>,
    metadata: Value,
) -> Result<PageEmbeddingBlob, Box<dyn std::error::Error + Send + Sync>> {
    let bytes = tokio::fs::read(path).await?;
    let stored = upsert_page_embedding_blob(
        pool,
        NewPageEmbeddingBlob {
            page_id,
            model_name: model_name.to_string(),
            model_revision: model_revision.map(str::to_string),
            embedding_shape,
            storage_format: "safetensors".to_string(),
            embedding_bytes: bytes,
            metadata,
        },
    )
    .await?;
    Ok(stored)
}

pub async fn get_page_embedding_blob(
    pool: &PgPool,
    page_id: Uuid,
    model_name: &str,
    model_revision: Option<&str>,
) -> Result<Option<PageEmbeddingBlob>, Error> {
    sqlx::query_as::<_, PageEmbeddingBlob>(
        r#"
        SELECT id, page_id, model_name, model_revision, embedding_shape,
               storage_format, embedding_bytes, metadata, created_at, updated_at
        FROM page_embedding_blobs
        WHERE page_id = $1
          AND model_name = $2
          AND (
              ($3::TEXT IS NULL AND model_revision IS NULL)
              OR model_revision = $3
          )
        "#,
    )
    .bind(page_id)
    .bind(model_name)
    .bind(model_revision)
    .fetch_optional(pool)
    .await
}
