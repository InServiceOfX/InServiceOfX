# vector_store

Rust PostgreSQL persistence layer for document-page embeddings.

This crate follows the GrokiCAD `database` pattern: `sqlx`, typed structs,
async `PgPool`, and integration tests that skip when Postgres is unavailable.

## Why Store ColQwen Embeddings As Blobs First

ColQwen embeddings are multi-vector tensors shaped roughly
`[num_patches, embedding_dim]`. Retrieval uses MaxSim across patch/token
vectors, not one cosine distance over a single vector. `pgvector` is still worth
enabling for future pooled-vector prefilters, but the ground-truth ColQwen
embedding should be stored as the original `.safetensors` bytes plus metadata.

The current Python indexer writes one `.safetensors` file per PDF page. This
crate can persist those files in PostgreSQL without trying to reinterpret the
tensor layout.

## Schema

See [sql/schema.sql](sql/schema.sql).

Core tables:

- `documents`
- `document_pages`
- `page_embedding_blobs`

## Local Test

Set `VECTOR_STORE_DATABASE_URL`, then run:

```bash
cd RustLibraries/vector_store
cargo test
```

If the env var is unset or the database is down, the integration test exits
successfully with a skip message.
