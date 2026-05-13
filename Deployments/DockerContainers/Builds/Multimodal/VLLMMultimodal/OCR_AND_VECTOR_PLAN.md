# OCR + Vector Persistence Plan

Last updated: 2026-05-12.

## Goals

1. Add two more document/OCR models to the multimodal comparison loop:
   - `/Data/Models/Multimodal/PaddlePaddle/PaddleOCR-VL-1.5`
   - `/Data/Models/Multimodal/zai-org/GLM-OCR`
2. Persist ColQwen page embeddings in PostgreSQL through Rust, not Python,
   following the GrokiCAD `sqlx` pattern.

## Current local model state

- `PaddlePaddle/PaddleOCR-VL-1.5`: local weights present, ~3.6 GB.
  The model card recommends the official PaddleOCR pipeline for page-level
  document parsing; it can delegate VLM recognition to a vLLM server.
- `zai-org/GLM-OCR`: local weights present, ~5.0 GB.
  The model card strongly recommends its official SDK for document parsing
  because it integrates PP-DocLayoutV3. vLLM serving requires nightly vLLM and
  a git/main transformers install per the card.
- `vidore/colqwen2.5-v0.2`: local LoRA adapter and base HF cache are present;
  Phase 4 app code exists but GPU end-to-end validation is pending.

## Important model-card constraints

### PaddleOCR-VL-1.5

Official card says:

- Install `paddlepaddle-gpu>=3.2.1` and `paddleocr[doc-parser]`.
- Basic path: `PaddleOCRVL().predict(...)`.
- Server-accelerated path: run a vLLM OpenAI-compatible server for the VLM
  component, then call `PaddleOCRVL(vl_rec_backend="vllm-server",
  vl_rec_server_url="http://127.0.0.1:8080/v1")`.
- The transformers example is element-level/text-spotting only, not full
  page-level document parsing.

Decision: prefer the official PaddleOCR CLI/Python API, optionally backed by
vLLM, because we care about full document parsing.

### GLM-OCR

Official card says:

- SDK is strongly recommended for document parsing because it integrates
  PP-DocLayoutV3 and structured output generation.
- vLLM path requires nightly vLLM and `transformers` from git.
- Prompt support is intentionally narrow: document parsing prompts like
  `Text Recognition:`, `Formula Recognition:`, `Table Recognition:` plus
  schema-driven information extraction.

Decision: current-image GLM-OCR is blocked. Use a separate deployment with
nightly vLLM and transformers main if we pursue it.

## Current image compatibility risk

`vllm-multimodal:25.06-py3` is pinned to:

- vLLM 0.11.2
- transformers 4.57.6
- torch 2.9.0

The OCR model cards point at newer or special stacks:

- PaddleOCR-VL's full official pipeline needs PaddlePaddle/PaddleOCR
  dependencies in addition to vLLM, but direct vLLM serving works in the
  current image when `--max-model-len 8192` is set.
- GLM-OCR vLLM instructions explicitly use nightly vLLM and transformers main.

So treat PaddleOCR-VL direct vLLM as available now, and GLM-OCR as a separate
Docker deployment task.

## Probe results on current image

### PaddleOCR-VL

Command shape tested:

```bash
vllm serve /Data/Models/Multimodal/PaddlePaddle/PaddleOCR-VL-1.5 \
  --served-model-name paddleocr-vl \
  --trust-remote-code \
  --port 8080 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.85 \
  --enforce-eager
```

Result: current vLLM 0.11.2 recognizes
`PaddleOCRVLForConditionalGeneration`, loads the ~3.6 GB model, and starts the
OpenAI-compatible API server on the RTX 3060. The default 131072-token context
is too heavy/slow for quick consumer-GPU probes; use `--max-model-len 8192`.

Implementation added:

- `PythonLibraries/ThirdParties/APIs/PaddleOCRVL/`
- `PythonApplications/CLIPDFPaddleOCRVLChat/`

This is direct VLM server usage, not the full official PaddleOCR pipeline.

### GLM-OCR

Command shape tested:

```bash
vllm serve /Data/Models/Multimodal/zai-org/GLM-OCR \
  --served-model-name glm-ocr \
  --trust-remote-code \
  --port 8080 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.85 \
  --enforce-eager \
  --allowed-local-media-path /
```

Result: current image fails before model load because transformers 4.57.6 does
not recognize `model_type: glm_ocr`. The model config advertises
`transformers_version: 5.0.1dev0`, matching the model-card requirement to
install transformers from source and use vLLM nightly.

Decision: do not mutate `VLLMMultimodal` for GLM-OCR yet. Create a separate OCR
deployment if we want to test it.

## Proposed Docker split

Avoid destabilizing the already-working MinerU/Qwen3VL/ColQwen image until
validated. Add OCR-specific components after the existing stack:

- `Dockerfile.paddleocr_vl`
  - install `paddlepaddle-gpu==3.2.1` for CUDA 12.x if compatible with the
    NVIDIA 25.06 base
  - install `paddleocr[doc-parser]`
  - import sanity check: `from paddleocr import PaddleOCRVL`
- `Dockerfile.glm_ocr`
  - only after probing whether current vLLM can load GLM-OCR
  - likely needs separate image or branch because nightly vLLM + transformers
    git can break MinerU/Qwen3VL compatibility

If GLM-OCR requires vLLM nightly, prefer a separate deployment such as
`Multimodal/OCRMultimodal` rather than mutating `VLLMMultimodal`.

## Python code location

These wrappers should not go under `PythonLibraries/HuggingFace` unless they
directly use HF `transformers`.

Use:

- `PythonLibraries/ThirdParties/APIs/PaddleOCRVL/`
- `PythonLibraries/ThirdParties/APIs/GLMOCR/`

Then add CLI apps under `PythonApplications/` for PDF processing, mirroring the
existing CLIPDF apps.

## Rust/PostgreSQL persistence plan

Use GrokiCAD as the template:

- `sqlx` with `runtime-tokio-rustls`, `postgres`, `json`, `uuid`, `chrono`
- small library crate with typed structs and async functions
- integration tests skip gracefully if DB is not running
- Docker Compose for local Postgres

Use InServiceOfX Python DB code as feature reference:

- create/list DBs
- create extension `vector`
- pool lifecycle

Implemented crate:

`RustLibraries/vector_store`

Implemented tables:

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE documents (
  id UUID PRIMARY KEY,
  source_path TEXT NOT NULL,
  source_sha256 TEXT,
  title TEXT,
  metadata JSONB DEFAULT '{}',
  created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(source_path, source_sha256)
);

CREATE TABLE document_pages (
  id UUID PRIMARY KEY,
  document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
  page_number INTEGER NOT NULL,
  image_path TEXT,
  width INTEGER,
  height INTEGER,
  metadata JSONB DEFAULT '{}',
  UNIQUE(document_id, page_number)
);

CREATE TABLE page_embedding_blobs (
  id UUID PRIMARY KEY,
  page_id UUID REFERENCES document_pages(id) ON DELETE CASCADE,
  model_name TEXT NOT NULL,
  model_revision TEXT,
  embedding_shape INTEGER[] NOT NULL,
  storage_format TEXT NOT NULL DEFAULT 'safetensors',
  embedding_bytes BYTEA NOT NULL,
  metadata JSONB DEFAULT '{}',
  created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(page_id, model_name, model_revision)
);
```

Note: ColQwen embeddings are multi-vector `[num_patches, dim]`, not a single
pgvector value. `pgvector` is ideal for single dense vectors, but ColQwen MaxSim
requires token/patch-level vectors. The implemented crate stores the full
`.safetensors` bytes plus shape metadata. Add derived pooled pgvector columns
later only if approximate prefiltering is useful.

Near-term path:

1. Keep the current `.safetensors` indexer as file-backed ground truth.
2. Use `RustLibraries/vector_store` to ingest those safetensors into Postgres.
3. Later add query-side retrieval from Postgres back into tensors for MaxSim.
