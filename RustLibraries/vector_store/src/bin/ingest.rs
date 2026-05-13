/// Ingest ColQwen page embeddings from a CLIPDFColQwenIndexer output directory
/// into a Postgres vector_store database.
///
/// Usage:
///   VECTOR_STORE_DATABASE_URL="postgres://..." cargo run --bin ingest -- \
///     --index-dir /path/to/CLIPDFColQwenIndexer/output \
///     [--model-name vidore/colqwen2.5-v0.2] \
///     [--dry-run]
///
/// The index directory must contain one or more subdirectories, each holding:
///   manifest.json  — written by CLIPDFColQwenIndexer
///   page_N.safetensors  — one file per indexed page

use std::path::PathBuf;

use serde::Deserialize;
use serde_json::json;
use vector_store::{
    create_pool, ensure_schema, upsert_document, upsert_document_page,
    store_safetensors_embedding_file, NewDocument, NewDocumentPage,
    DEFAULT_MODEL_NAME_COLQWEN2_5_V0_2,
};

#[derive(Debug, Deserialize)]
struct Manifest {
    pdf_path: String,
    pdf_stem: String,
    num_pages: usize,
    dpi: u32,
    pages: Vec<PageEntry>,
}

#[derive(Debug, Deserialize)]
struct PageEntry {
    page: i32,
    status: String,
    embedding: Option<String>,
    embedding_shape: Option<Vec<i32>>,
    image: Option<String>,
    #[serde(default)]
    seconds: f64,
}

#[derive(Debug)]
struct Args {
    index_dir: PathBuf,
    model_name: String,
    model_revision: Option<String>,
    dry_run: bool,
}

fn parse_args() -> Args {
    let mut index_dir: Option<PathBuf> = None;
    let mut model_name = DEFAULT_MODEL_NAME_COLQWEN2_5_V0_2.to_string();
    let mut model_revision: Option<String> = None;
    let mut dry_run = false;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--index-dir" => {
                index_dir = args.next().map(PathBuf::from);
            }
            "--model-name" => {
                if let Some(v) = args.next() {
                    model_name = v;
                }
            }
            "--model-revision" => {
                model_revision = args.next();
            }
            "--dry-run" => {
                dry_run = true;
            }
            "--help" | "-h" => {
                eprintln!(
                    "Usage: VECTOR_STORE_DATABASE_URL=... ingest \
                     --index-dir <path> [--model-name <name>] [--model-revision <rev>] [--dry-run]"
                );
                std::process::exit(0);
            }
            other => {
                eprintln!("Unknown argument: {other}");
                std::process::exit(1);
            }
        }
    }

    Args {
        index_dir: index_dir.unwrap_or_else(|| {
            eprintln!("--index-dir is required");
            std::process::exit(1);
        }),
        model_name,
        model_revision,
        dry_run,
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let args = parse_args();

    let database_url = std::env::var("VECTOR_STORE_DATABASE_URL").unwrap_or_else(|_| {
        eprintln!("VECTOR_STORE_DATABASE_URL must be set");
        std::process::exit(1);
    });

    if args.dry_run {
        println!("[dry-run] Would connect to: {database_url}");
    }

    let pool = if args.dry_run {
        None
    } else {
        println!("Connecting to Postgres…");
        let p = create_pool(&database_url).await?;
        ensure_schema(&p).await?;
        println!("Schema ready.");
        Some(p)
    };

    let index_dir = &args.index_dir;
    if !index_dir.exists() {
        eprintln!("index_dir does not exist: {}", index_dir.display());
        std::process::exit(1);
    }

    // Collect subdirectories that have a manifest.json
    let mut doc_dirs: Vec<PathBuf> = Vec::new();
    if index_dir.join("manifest.json").exists() {
        doc_dirs.push(index_dir.clone());
    } else {
        let mut entries: Vec<_> = std::fs::read_dir(index_dir)?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().is_dir())
            .collect();
        entries.sort_by_key(|e| e.path());
        for entry in entries {
            let manifest = entry.path().join("manifest.json");
            if manifest.exists() {
                doc_dirs.push(entry.path());
            }
        }
    }

    if doc_dirs.is_empty() {
        eprintln!("No manifest.json found under {}", index_dir.display());
        std::process::exit(1);
    }

    println!(
        "Found {} document director{}.",
        doc_dirs.len(),
        if doc_dirs.len() == 1 { "y" } else { "ies" }
    );

    let mut total_pages_ingested = 0usize;
    let mut total_pages_skipped = 0usize;
    let mut total_errors = 0usize;

    for doc_dir in &doc_dirs {
        let manifest_text = std::fs::read_to_string(doc_dir.join("manifest.json"))?;
        let manifest: Manifest = serde_json::from_str(&manifest_text)?;

        println!("\n[{}]  {} pages", manifest.pdf_stem, manifest.num_pages);

        let doc_record = if let Some(ref p) = pool {
            let doc = upsert_document(
                p,
                NewDocument {
                    source_path: manifest.pdf_path.clone(),
                    source_sha256: None,
                    title: Some(manifest.pdf_stem.clone()),
                    metadata: json!({
                        "dpi": manifest.dpi,
                        "index_dir": doc_dir.to_string_lossy(),
                    }),
                },
            )
            .await?;
            Some(doc)
        } else {
            None
        };

        for page_entry in &manifest.pages {
            if page_entry.status != "ok" {
                println!("  page {:>3}: skipped (status={})", page_entry.page, page_entry.status);
                total_pages_skipped += 1;
                continue;
            }

            let embedding_name = match &page_entry.embedding {
                Some(name) => name.clone(),
                None => {
                    println!("  page {:>3}: skipped (no embedding field)", page_entry.page);
                    total_pages_skipped += 1;
                    continue;
                }
            };

            let embedding_path = doc_dir.join(&embedding_name);
            if !embedding_path.exists() {
                eprintln!("  page {:>3}: ERROR — embedding file not found: {}", page_entry.page, embedding_path.display());
                total_errors += 1;
                continue;
            }

            let shape = page_entry.embedding_shape.clone().unwrap_or_default();

            if args.dry_run {
                println!(
                    "  page {:>3}: would ingest {} shape={:?}",
                    page_entry.page,
                    embedding_name,
                    shape,
                );
                total_pages_ingested += 1;
                continue;
            }

            let p = pool.as_ref().unwrap();
            let doc_id = doc_record.as_ref().unwrap().id;
            let image_path_str = page_entry.image.as_ref().map(|img| {
                doc_dir.join(img).to_string_lossy().to_string()
            });

            let page_record = upsert_document_page(
                p,
                NewDocumentPage {
                    document_id: doc_id,
                    page_number: page_entry.page,
                    image_path: image_path_str,
                    width: None,
                    height: None,
                    metadata: json!({"seconds": page_entry.seconds}),
                },
            )
            .await?;

            match store_safetensors_embedding_file(
                p,
                page_record.id,
                &args.model_name,
                args.model_revision.as_deref(),
                shape.clone(),
                &embedding_path,
                json!({}),
            )
            .await
            {
                Ok(_) => {
                    println!("  page {:>3}: ingested  {} shape={:?}", page_entry.page, embedding_name, shape);
                    total_pages_ingested += 1;
                }
                Err(e) => {
                    eprintln!("  page {:>3}: ERROR — {e}", page_entry.page);
                    total_errors += 1;
                }
            }
        }
    }

    println!(
        "\nDone. ingested={total_pages_ingested} skipped={total_pages_skipped} errors={total_errors}"
    );

    Ok(())
}
