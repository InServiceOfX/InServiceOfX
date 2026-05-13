# PIDViewer

FastAPI web server for inspecting P&ID extraction output from `CLIPDFExtraction`
(MinerU) and `CLIPDFQwen3VLChat` (Qwen3-VL). Runs on the host — no GPU needed.

## What it shows

- All documents found under the configured MinerU output directory, as a
  three-panel layout: page strip / page image / extraction content.
- Per-page structured MinerU elements (type-tagged, with bbox).
- Per-page Qwen3VL free-text extraction.
- Per-page stats (element count, element types, extraction time).
- Whether a ColQwen embedding index exists for each page.

## Running on the host

1. Install deps once (into the existing InServiceOfX venv):
   ```bash
   .venv/bin/python -m pip install fastapi "uvicorn[standard]" aiofiles pydantic pyyaml
   ```

2. Copy the example config:
   ```bash
   cp Configurations/viewer_configuration.yml.example Configurations/viewer_configuration.yml
   # Edit paths to match your host output directories
   ```

3. Start:
   ```bash
   python Executables/main_PIDViewer.py
   # → http://localhost:8888
   ```

## Running inside the Docker container

Mount the `PIDViewer` directory into the container alongside `InServiceOfX` and
install the deps. Paths in `viewer_configuration.yml` should use container paths
(`/Workspace/Generated/...`).

## API endpoints

| Endpoint | Description |
| --- | --- |
| `GET /` | Serves the single-page HTML viewer |
| `GET /api/status` | Server config + document count |
| `GET /api/documents` | List all document IDs |
| `GET /api/documents/{doc}` | Document metadata + per-page summary |
| `GET /api/documents/{doc}/pages/{page}/image` | PNG raster |
| `GET /api/documents/{doc}/pages/{page}/mineru` | MinerU JSON elements |
| `GET /api/documents/{doc}/pages/{page}/qwen3vl` | Qwen3VL text |

`{doc}` is the subdirectory name under `mineru_output_path`; URL-encode spaces.

## Quality observations on dense multi-page P&IDs

- **MinerU** (page 3): produced a hallucinated Mermaid flowchart as the `image`
  element content — the model invented a long sequential `LOX Supply → LOX Supply`
  chain that does not match the actual P&ID topology. Header/footer/table
  elements on simpler pages are accurate.
- **Qwen3VL** (page 3): hallucinated a sequential `S2TYP1 → S2TYP100+` tag chain
  — a known failure mode on dense P&IDs. Page 1 (title block only) was accurate.

Both models produce unreliable connectivity on dense full-page diagrams. See
`OCR_AND_VECTOR_PLAN.md` for the tiled extraction roadmap.
