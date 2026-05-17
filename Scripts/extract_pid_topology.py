"""
P&ID topology extractor using Claude API (claude-sonnet-4-6).

Reads page_manifest.json files produced by classify_pid_pages.py, sends each
"pid"-labelled page image to Claude, and writes:
  - topology.mermaid   — Mermaid flowchart LR
  - topology.json      — structured adjacency list for programmatic use
  - topology_raw.txt   — Claude's full response (for debugging)

Usage:
    python /InServiceOfX/Scripts/extract_pid_topology.py [--doc DOC_SUBDIR] [--model MODEL]

    DOC_SUBDIR: subdirectory under /Workspace/Generated/CLIPDFExtraction/
                (also used to find the manifest under /Workspace/Generated/PIDManifest/)
    If omitted, processes all docs with a manifest.

Requirements:
    pip install anthropic           (already present on the host, run on host or pass key)
    ANTHROPIC_API_KEY env var must be set.

Output: /Workspace/Generated/PIDTopology/<doc_name>/page_<n>/
"""

import argparse
import base64
import json
import os
import sys
from pathlib import Path

MANIFEST_ROOT = Path("/Workspace/Generated/PIDManifest")
OUTPUT_ROOT   = Path("/Workspace/Generated/PIDTopology")

DEFAULT_MODEL = "claude-sonnet-4-6"

SYSTEM_PROMPT = """\
You are an aerospace systems engineer specializing in Piping and Instrumentation \
Diagrams (P&IDs). You extract complete, accurate flow topologies from P&ID images \
and represent them in two formats simultaneously.

Rules for extraction:
- Include EVERY component visible: tanks, valves (manual, solenoid, check, relief, \
regulator), instruments (pressure gauges, transducers, flow meters, temperature \
sensors), filters, heat exchangers, vents, fill/drain ports, and T-junctions.
- Use the component's tag identifier exactly as printed (e.g. S2BHFB, GN2-1, PG-101).
  If no tag is visible, use a descriptive label (e.g. "LOX Tank", "Check Valve").
- Preserve flow direction (arrow direction = fluid flow direction).
- Label edges with the fluid/medium when shown (LOX, GHe, GN2, RP-1, etc.).
- T-junctions and manifolds are nodes with multiple outgoing or incoming edges.
- Cross-page references ("Cont. on Sh. 2") should be noted as boundary nodes.

Mermaid node shape conventions:
  - Tanks / vessels:           [(Label)]       cylinder
  - Manual valves:             [Label]         rectangle
  - Solenoid / actuated valves:[Label]         rectangle  (add "SV" prefix if missing)
  - Check valves:              [Label]         rectangle
  - Regulators:                [Label]         rectangle
  - Pressure gauges / sensors: ((Label))       circle
  - Flow inputs / outputs:     [/Label/]       parallelogram
  - Vent / atmosphere:         ([Label])       stadium
  - T-junction / tee:          {Label}         rhombus

Output format — reply with EXACTLY two fenced code blocks, nothing else:

```mermaid
flowchart LR
    ...
```

```json
{
  "nodes": [
    {"id": "node_id", "tag": "S2BHFB", "type": "solenoid_valve", "label": "S2BHFB\\nLOX Bleed Valve"}
  ],
  "edges": [
    {"from": "node_id_a", "to": "node_id_b", "medium": "LOX", "label": ""}
  ]
}
```

Node types for JSON: tank, manual_valve, solenoid_valve, check_valve, regulator,
pressure_gauge, pressure_transducer, flow_meter, temperature_sensor, filter,
heat_exchanger, orifice, relief_valve, vent, fill_drain, tee, boundary, other.
"""

USER_PROMPT = "Extract the complete P&ID flow topology from this diagram."


def image_to_base64(path: Path) -> str:
    return base64.standard_b64encode(path.read_bytes()).decode("utf-8")


def call_claude(client, image_path: Path, model: str) -> str:
    img_b64 = image_to_base64(image_path)
    message = client.messages.create(
        model=model,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": img_b64,
                        },
                    },
                    {"type": "text", "text": USER_PROMPT},
                ],
            }
        ],
    )
    return message.content[0].text


def parse_response(raw: str) -> tuple[str, dict | None]:
    """Extract mermaid and json blocks from Claude's response."""
    import re
    mermaid = ""
    topology_json = None

    mermaid_match = re.search(r"```mermaid\n(.*?)```", raw, re.DOTALL)
    if mermaid_match:
        mermaid = mermaid_match.group(1).strip()

    json_match = re.search(r"```json\n(.*?)```", raw, re.DOTALL)
    if json_match:
        try:
            topology_json = json.loads(json_match.group(1))
        except json.JSONDecodeError as e:
            print(f"    JSON parse error: {e}")

    return mermaid, topology_json


def process_manifest(manifest_path: Path, client, model: str) -> None:
    data = json.loads(manifest_path.read_text())
    doc_name = data["document"]
    pid_pages = [p for p in data["pages"] if p["label"] == "pid"]

    if not pid_pages:
        print(f"  No P&ID pages in {doc_name}, skipping.")
        return

    print(f"  {len(pid_pages)} P&ID page(s): {[p['page'] for p in pid_pages]}")

    for page_info in pid_pages:
        page_num = page_info["page"]
        image_path = Path(page_info["path"])

        if not image_path.exists():
            print(f"    page {page_num}: image not found at {image_path}, skipping.")
            continue

        out_dir = OUTPUT_ROOT / doc_name / f"page_{page_num}"
        out_dir.mkdir(parents=True, exist_ok=True)

        raw_file   = out_dir / "topology_raw.txt"
        mermaid_file = out_dir / "topology.mermaid"
        json_file  = out_dir / "topology.json"

        # Skip if already done
        if mermaid_file.exists() and json_file.exists():
            print(f"    page {page_num}: already extracted, skipping.")
            continue

        print(f"    page {page_num}: calling Claude ({model})...", flush=True)
        raw = call_claude(client, image_path, model)
        raw_file.write_text(raw)

        mermaid, topology_json = parse_response(raw)

        if mermaid:
            mermaid_file.write_text(mermaid)
            print(f"    page {page_num}: Mermaid written ({len(mermaid)} chars, "
                  f"{mermaid.count('-->') + mermaid.count('---')} edges)")
        else:
            print(f"    page {page_num}: WARNING — no Mermaid block in response")
            mermaid_file.write_text(f"% No mermaid block extracted\n% Raw response:\n% {raw[:200]}")

        if topology_json:
            json_file.write_text(json.dumps(topology_json, indent=2))
            n_nodes = len(topology_json.get("nodes", []))
            n_edges = len(topology_json.get("edges", []))
            print(f"    page {page_num}: JSON written ({n_nodes} nodes, {n_edges} edges)")
        else:
            print(f"    page {page_num}: WARNING — no JSON block in response")
            json_file.write_text("{}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc", default=None,
                        help="Single doc name (manifest subdirectory)")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Claude model ID (default: {DEFAULT_MODEL})")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set.", file=sys.stderr)
        sys.exit(1)

    try:
        import anthropic
    except ImportError:
        print("ERROR: 'anthropic' package not installed. Run: pip install anthropic",
              file=sys.stderr)
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    if args.doc:
        manifests = [MANIFEST_ROOT / args.doc / "page_manifest.json"]
    else:
        manifests = sorted(MANIFEST_ROOT.rglob("page_manifest.json"))

    if not manifests:
        print("No manifests found. Run classify_pid_pages.py first.")
        sys.exit(1)

    for manifest_path in manifests:
        if not manifest_path.exists():
            print(f"Manifest not found: {manifest_path}")
            continue
        print(f"\nDocument: {manifest_path.parent.name}")
        process_manifest(manifest_path, client, args.model)

    print("\nDone. Results in:", OUTPUT_ROOT)


if __name__ == "__main__":
    main()
