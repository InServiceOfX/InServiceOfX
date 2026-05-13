# NEXT_STEPS.md — open backlog for VLLMMultimodal

Concrete, scoped tasks. Each is sized for a single agent session (~1-3 hours). Pick the lowest-numbered open task. Mark completed tasks by moving the heading from `## OPEN` to `## DONE` (newest first) and leaving the body in place as historical record.

The current image (`vllm-multimodal:25.06-py3` sha `f96c8b89ae50`) supports MinerU, Qwen3-VL, ColQwen, and direct PaddleOCR-VL serving. GLM-OCR does not work in this image because it needs a newer/nightly vLLM + transformers stack.

As of 2026-05-12, the ColQwen index/query app code, PaddleOCR-VL direct API app, and Rust Postgres vector store crate exist. The next work is validation and integration, not first-pass scaffolding. See `HANDOFF_2026-05-12.md` before starting.

## OPEN

### 1. GPU-validate `CLIPDFColQwenIndexer`

**Implementation note (2026-05-11):** app code and `.example` configs now exist
under `PythonApplications/CLIPDFColQwenIndexer/`; host syntax/import checks pass.
Still open until the GPU end-to-end run produces embeddings + manifest for the
target P&ID corpus and the round-trip check below is done.

**Goal.** Given a directory of PDFs and a config, rasterize each page, embed it via `ColQwen2_5Embedder`, and persist the resulting multi-vector embeddings to disk. Subsequent queries can MaxSim against the index without re-embedding.

**Mirrors.** `PythonApplications/CLIPDFExtraction/` and `PythonApplications/CLIPDFQwen3VLChat/` — same shape, different runner.

**Files added** under `PythonApplications/CLIPDFColQwenIndexer/`:
- `pyproject.toml`, `README.md` (copy from CLIPDFQwen3VLChat, change names)
- `Configurations/colqwen2_5_configuration.yml.example` — model_path, torch_dtype, device_map (mirror the wrapper config; we already have one usable example in `CLIPDFQwen3VLChat/Configurations/qwen3vl_configuration.yml.example` to adapt)
- `Configurations/pdf_index_configuration.yml.example` — input_path (dir), output_path, pdf_dpi, image_format, save_intermediate_images, skip_existing, **`embeddings_format: safetensors`** (or `.pt` — pick one and document)
- `Executables/main_CLIPDFColQwenIndexer.py`
- `clipdfcolqwenindexer/` package with:
  - `__init__.py` exporting `ApplicationPaths`
  - `ApplicationPaths.py` (mirror, change configuration_file_paths keys to `colqwen2_5_configuration` + `pdf_index_configuration`)
  - `CLIPDFColQwenIndexer.py` (mirror; import `ColQwen2_5Embedder`)
  - `Core/__init__.py`
  - `Core/ProcessConfigurations.py` (loads both YAMLs)
  - `Core/PDFIndexConfiguration.py` (pydantic; add `embeddings_format` field, no `prompt` field since we're indexing not asking)
  - `Core/PDFRasterizer.py` (local copy of the one from CLIPDFExtraction — they're identical)
  - `Core/IndexRunner.py` — analog to `ChatRunner` but writes embeddings instead of text. Output layout suggestion:
    ```
    <output_path>/<pdf_stem>/
      embeddings.safetensors        # shape [num_pages, num_patches, embed_dim]
      page_N.png                    # if save_intermediate_images
      manifest.json                 # {pdf_path, num_pages, dpi, model_path, sha_of_lora, pages: [{page, status, embedding_index, seconds}]}
    ```

**Notes for the implementer:**
- The wrapper's `embed_images([pil1, pil2, ...])` returns one multi-vector tensor per image. Stack them per-PDF; persist via `safetensors.torch.save_file({"embeddings": tensor, "page_indices": tensor([1, 2, ...])}, path)`.
- `num_patches` varies per image. Either pad to max (with a mask) OR store each page's embedding as a separate file. Pick the per-page-file approach for simplicity in v1.
- Update `.gitignore` with `PythonApplications/CLIPDFColQwenIndexer/Configurations/*.yml`.
- Add to root `STATUS.md` once done.

**Definition of done.**
1. `python Executables/main_CLIPDFColQwenIndexer.py --currentpath` on the example PDFs dir produces one subdir per PDF with persisted embeddings + manifest.
2. Manifest's `status: ok` for every page.
3. The embeddings tensor for any page round-trips: load it back, compare to `embed_images([page])` cosine — should be identical or near-identical (allowing for AMP nondeterminism).

### 2. GPU-validate `CLIPDFColQwenQuery`

**Implementation note (2026-05-11):** app code and `.example` configs now exist
under `PythonApplications/CLIPDFColQwenQuery/`; host syntax/import checks pass.
Still open until task #1 has a real index and this query CLI is GPU-tested.

**Goal.** Given an index produced by task #1 + a text query, return the top-K page hits across the entire corpus, ranked by MaxSim score.

**Files added** under `PythonApplications/CLIPDFColQwenQuery/`:
- Same shape as task #1.
- `Configurations/colqwen2_5_configuration.yml.example` — identical to task #1's.
- `Configurations/pdf_query_configuration.yml.example` — `index_path` (output dir from #1), `query` (string OR list of strings), `top_k` (int, default 5), `output_path` (where to write the result JSON).
- `Executables/main_CLIPDFColQwenQuery.py`
- `clipdfcolqwenquery/Core/QueryRunner.py`:
  - Walks `index_path` for all per-page embedding files, builds an in-memory list with (pdf_stem, page_index, embeddings_tensor)
  - Embeds the query via `embed_queries([query_str])`
  - For each candidate, computes MaxSim via the wrapper's `score(query_embeddings, page_embeddings)` and concatenates
  - Sorts descending, returns top-K
  - Writes `query_result.json`:
    ```json
    {"query": "...", "top_k": 5, "hits": [{"rank": 1, "pdf": "...", "page": 4, "score": 16.12, "page_png": "..."}, ...]}
    ```

**Definition of done.**
1. End-to-end on the example corpus: index with task #1, query "show me valve specifications", get a top-5 hit list that's plausibly ordered (engineers can sanity-check it).
2. Re-running with the same query is reproducible (greedy / no randomness).
3. Query latency for a corpus of ~80 pages should be sub-second after model load.

### 3. Wire `RustLibraries/vector_store` to ColQwen index outputs

**Goal.** Ingest the indexer's per-page `.safetensors` files into PostgreSQL so
the exact ColQwen multi-vector tensors can persist outside the filesystem.

**Existing code.**

- `RustLibraries/vector_store/src/lib.rs`
- `RustLibraries/vector_store/sql/schema.sql`
- `RustLibraries/vector_store/tests/integration.rs`

**Definition of done.**

1. Start or point at a Postgres database with `pgvector` available.
2. Set `VECTOR_STORE_DATABASE_URL=postgres://...`.
3. Run `cargo test` and confirm the integration test performs real DB I/O
   instead of skipping.
4. Add a small CLI or ingestion example that walks a
   `CLIPDFColQwenIndexer` output directory and calls
   `store_safetensors_embedding_file`.

### 4. Run `CLIPDFPaddleOCRVLChat` on a target PDF

**Goal.** Produce comparable PaddleOCR-VL page outputs for the same PDF used by
MinerU and Qwen3-VL.

**Server command.**

```bash
vllm serve /Data/Models/Multimodal/PaddlePaddle/PaddleOCR-VL-1.5 \
  --served-model-name paddleocr-vl \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 8080 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.85 \
  --enforce-eager
```

**Caution.** The API smoke on page 3 returned an invented-looking sequential
tag list (`S2N001...S2N142`). Evaluate these outputs as model-comparison data,
not trusted P&ID connectivity.

### 5. Design tiled P&ID extraction

Full-page VLM prompting is not reliable enough for dense P&ID connectivity.
Next architecture should combine:

- high-DPI rasterization
- overlapping tiles
- exact tag OCR per tile
- table/title-block cross-checks
- geometry/vector analysis of lines, endpoints, arrows, and symbols
- local VLM explanations constrained by OCR + geometry facts

### 6. GLM-OCR separate deployment

Only do this if Ernest wants to continue with GLM-OCR. Current
`VLLMMultimodal` fails because `transformers==4.57.6` does not recognize
`glm_ocr`. The GLM-OCR model card points at nightly vLLM and transformers main,
so create a separate deployment rather than destabilizing the current image.

### 7. CLI quality-of-life polish (low priority)

If both #1 and #2 are done and the agent has cycles:

- A combined `CLIPDFColQwenRetriever` with subcommands (`index`, `query`) instead of two separate apps. Move the duplicated PDFRasterizer to a shared library.
- A `--limit-pages N` flag on the indexer (for fast iteration during development).
- A `--combine-queries` flag on the query CLI: given multiple queries, return the top-K for each plus the top-K of the *intersection* score.

## DONE (newest first)

### 2026-05-12 — scaffolding and probes

- `CLIPDFColQwenIndexer` app code and `.example` configs added.
- `CLIPDFColQwenQuery` app code and `.example` configs added.
- `RustLibraries/vector_store` crate added and `cargo test` passes.
- `PaddleOCRVL` direct API wrapper and `CLIPDFPaddleOCRVLChat` app added.
- PaddleOCR-VL direct vLLM server smoke passed.
- GLM-OCR current-image probe failed as expected due `glm_ocr` transformer support.

Phase 1, 2, 3 wrappers + smoke tests + CLI apps for Phases 1+2 are complete. Phase 4 app code exists; GPU CLI validation remains open. See `STATUS.md`.
