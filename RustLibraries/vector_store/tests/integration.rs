use serde_json::json;
use vector_store::{
    create_pool, ensure_schema, get_page_embedding_blob, store_safetensors_embedding_file,
    upsert_document, upsert_document_page, NewDocument, NewDocumentPage,
    DEFAULT_MODEL_NAME_COLQWEN2_5_V0_2,
};

#[tokio::test]
async fn store_and_fetch_embedding_blob() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let Ok(database_url) = std::env::var("VECTOR_STORE_DATABASE_URL") else {
        eprintln!("VECTOR_STORE_DATABASE_URL unset; skipping integration test.");
        return Ok(());
    };

    let pool = match create_pool(&database_url).await {
        Ok(pool) => pool,
        Err(error) => {
            eprintln!("Could not connect to Postgres ({error}); skipping integration test.");
            return Ok(());
        }
    };

    ensure_schema(&pool).await?;

    let document = upsert_document(
        &pool,
        NewDocument {
            source_path: "test://sample-pid.pdf".to_string(),
            source_sha256: Some("integration-test".to_string()),
            title: Some("Sample P&ID Document".to_string()),
            metadata: json!({"test": true}),
        },
    )
    .await?;

    let page = upsert_document_page(
        &pool,
        NewDocumentPage {
            document_id: document.id,
            page_number: 3,
            image_path: Some("/tmp/page_3.png".to_string()),
            width: Some(100),
            height: Some(200),
            metadata: json!({"kind": "pid"}),
        },
    )
    .await?;

    let temp_dir = std::env::temp_dir();
    let embedding_path = temp_dir.join(format!("vector-store-{}.safetensors", page.id));
    tokio::fs::write(&embedding_path, b"fake-safetensors-bytes").await?;

    let stored = store_safetensors_embedding_file(
        &pool,
        page.id,
        DEFAULT_MODEL_NAME_COLQWEN2_5_V0_2,
        Some("test-revision"),
        vec![768, 128],
        &embedding_path,
        json!({"source": "integration"}),
    )
    .await?;
    assert_eq!(stored.embedding_shape, vec![768, 128]);
    assert_eq!(stored.embedding_bytes, b"fake-safetensors-bytes");

    let fetched = get_page_embedding_blob(
        &pool,
        page.id,
        DEFAULT_MODEL_NAME_COLQWEN2_5_V0_2,
        Some("test-revision"),
    )
    .await?;
    assert!(fetched.is_some());
    assert_eq!(fetched.unwrap().embedding_bytes, b"fake-safetensors-bytes");

    tokio::fs::remove_file(&embedding_path).await.ok();

    sqlx::query("DELETE FROM documents WHERE id = $1")
        .bind(document.id)
        .execute(&pool)
        .await?;

    Ok(())
}
