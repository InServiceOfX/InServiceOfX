"""
P&ID page classifier using ColQwen2.5.

Encodes all pages of one or more extracted PDF documents, queries with
"piping and instrumentation diagram", and writes a page_manifest.json that
labels each page as "pid" or "supporting" based on similarity score.

Usage (inside vllm-multimodal container):
    python /InServiceOfX/Scripts/classify_pid_pages.py [--doc DOC_SUBDIR] [--threshold 0.15]

    DOC_SUBDIR is a subdirectory name under /Workspace/Generated/CLIPDFExtraction/.
    If omitted, all documents are processed.

Output: /Workspace/Generated/PIDManifest/<doc_name>/page_manifest.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from PIL import Image

EXTRACT_ROOT = Path("/Workspace/Generated/CLIPDFExtraction")
OUTPUT_ROOT  = Path("/Workspace/Generated/PIDManifest")
MODEL_PATH   = "/Data/Models/Multimodal/vidore/colqwen2.5-v0.2"

# Queries: average scores across multiple phrasings for robustness.
QUERIES = [
    "piping and instrumentation diagram",
    "P&ID process flow schematic valves instruments",
    "propellant pressurant fluid system diagram",
]


def load_model():
    from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
    print("Loading ColQwen2.5...", flush=True)
    model = ColQwen2_5.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        local_files_only=True,
    ).eval()
    processor = ColQwen2_5_Processor.from_pretrained(MODEL_PATH, local_files_only=True)
    print("ColQwen2.5 ready.", flush=True)
    return model, processor


@torch.no_grad()
def encode_images(model, processor, image_paths: list[Path]) -> torch.Tensor:
    images = [Image.open(p).convert("RGB") for p in image_paths]
    batch = processor.process_images(images).to(model.device)
    embeddings = model(**batch)          # (N, seq_len, dim) multi-vector
    return embeddings                    # stays on GPU


@torch.no_grad()
def encode_queries(model, processor, queries: list[str]) -> torch.Tensor:
    batch = processor.process_queries(queries).to(model.device)
    return model(**batch)                # (Q, seq_len, dim)


def maxsim_score(doc_emb: torch.Tensor, query_emb: torch.Tensor) -> float:
    """
    ColBERT-style MaxSim: for each query token, find its max similarity
    to any document token; sum across query tokens.  Average over queries.
    """
    # doc_emb: (N, Sd, D), query_emb: (Q, Sq, D)
    # Compute per-document per-query score
    scores = []
    for q in query_emb:                  # q: (Sq, D)
        # sim: (N, Sq, Sd)
        sim = torch.einsum("sd,npd->nsp", q, doc_emb)
        max_sim = sim.max(dim=-1).values  # (N, Sq)
        score = max_sim.sum(dim=-1)       # (N,)
        scores.append(score)
    return torch.stack(scores).mean(dim=0)   # (N,)


def classify_document(doc_dir: Path, model, processor, threshold: float) -> dict:
    page_pngs = sorted(doc_dir.glob("page_*.png"),
                       key=lambda p: int(p.stem.split("_")[1]))
    if not page_pngs:
        print(f"  No page_*.png found in {doc_dir}, skipping.")
        return {}

    print(f"  Encoding {len(page_pngs)} pages...", flush=True)
    doc_embs = encode_images(model, processor, page_pngs)   # (N, Sd, D)

    print(f"  Encoding {len(QUERIES)} queries...", flush=True)
    q_embs = encode_queries(model, processor, QUERIES)       # (Q, Sq, D)

    scores = maxsim_score(doc_embs, q_embs).cpu().tolist()   # list[float]

    # Normalise to [0,1] range for readability
    min_s, max_s = min(scores), max(scores)
    span = max_s - min_s if max_s > min_s else 1.0
    norm = [(s - min_s) / span for s in scores]

    pages = []
    for i, (png, raw, n) in enumerate(zip(page_pngs, scores, norm), start=1):
        label = "pid" if n >= threshold else "supporting"
        pages.append({
            "page":       i,
            "path":       str(png),
            "raw_score":  round(raw, 4),
            "norm_score": round(n, 4),
            "label":      label,
        })
        print(f"    page {i:2d}  norm={n:.3f}  → {label}")

    return {
        "document": doc_dir.name,
        "threshold": threshold,
        "queries": QUERIES,
        "pages": pages,
    }


# Map container path prefix → host path prefix so the prompt files contain
# paths that Claude Code (running on the host) can read directly.
CONTAINER_EXTRACT_PREFIX = str(EXTRACT_ROOT)
HOST_EXTRACT_PREFIX = "/home/propdev/.openclaw/workspace/workspace2/Data/Generated/CLIPDFExtraction"


def write_claude_prompt(out_dir: Path, result: dict) -> None:
    pid_pages = [p for p in result["pages"] if p["label"] == "pid"]
    if not pid_pages:
        return

    lines = [
        "Please extract the P&ID flow topology from each of the following diagram pages.",
        "For each page, use the Read tool on the image file path listed and produce:",
        "  1. A mermaid code block -- complete flowchart LR with ALL components and connections",
        "  2. A json code block -- adjacency list: nodes (id, tag, type, label) and edges (from, to, medium, label)",
        "",
        "Node shape conventions for Mermaid:",
        "  Tanks/vessels: [(Label)]   Manual/solenoid/check valves, regulators: [Label]",
        "  Gauges/sensors: ((Label))  Flow in/out: [/Label/]   Vents: ([Label])   T-junctions: {Label}",
        "",
        "Include EVERY component with its tag identifier exactly as printed.",
        "Label edges with fluid/medium (LOX, GHe, GN2, RP-1, He, etc.).",
        "For multi-page P&IDs, note cross-page references as boundary nodes.",
        "Output one pair of code blocks per page, preceded by: ## Page N -- <doc name>",
        "", "---", "",
    ]

    for p in pid_pages:
        host_path = p["path"].replace(CONTAINER_EXTRACT_PREFIX, HOST_EXTRACT_PREFIX)
        lines.append(f"Page {p['page']} (ColQwen score {p['norm_score']:.3f}):")
        lines.append(f"  {host_path}")
        lines.append("")

    prompt_file = out_dir / "claude_code_prompt.txt"
    prompt_file.write_text("\n".join(lines))
    print(f"  Claude Code prompt: {prompt_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc", default=None,
                        help="Single doc subdirectory name under CLIPDFExtraction")
    parser.add_argument("--threshold", type=float, default=0.40,
                        help="Normalised score threshold for 'pid' label (0-1)")
    args = parser.parse_args()

    model, processor = load_model()

    if args.doc:
        doc_dirs = [EXTRACT_ROOT / args.doc]
    else:
        doc_dirs = sorted(d for d in EXTRACT_ROOT.iterdir() if d.is_dir())

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    for doc_dir in doc_dirs:
        print(f"\nProcessing: {doc_dir.name}", flush=True)
        result = classify_document(doc_dir, model, processor, args.threshold)
        if not result:
            continue

        out_dir = OUTPUT_ROOT / doc_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "page_manifest.json"
        out_file.write_text(json.dumps(result, indent=2))
        print(f"  Manifest written: {out_file}")

        pid_pages = [p for p in result["pages"] if p["label"] == "pid"]
        print(f"  P&ID pages ({len(pid_pages)}): {[p['page'] for p in pid_pages]}")

        write_claude_prompt(out_dir, result)

    print("\nDone.")


if __name__ == "__main__":
    main()
