"""Two-stage P&ID topology extractor using Qwen3-VL.

Stage 1: extract nodes (components + tag numbers) as JSON
Stage 2: extract edges (flow connections + directions) as JSON
Stage 3: synthesize Mermaid flowchart from JSON (pure Python)

Run from inside the VLLMMultimodal container:

    cd /InServiceOfX/PythonApplications/CLIPDFQwen3VLChat
    python Executables/main_pid_topology.py \\
        --image /Workspace/Public/Space/PAndID/extracted/psas_pid-20.png \\
        --output /Workspace/Public/Space/PAndID/extracted/psas_pid-20_topology \\
        --configpath /InServiceOfX/PythonApplications/CLIPDFQwen3VLChat

Results written to <output>/:
    nodes.json          — VLM node extraction (Stage 1 raw response)
    edges.json          — VLM edge extraction (Stage 2 raw response)
    topology.json       — parsed node+edge graph
    topology.mermaid    — Mermaid flowchart LR
    report.txt          — full run log with timings
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


# ── Path bootstrap ────────────────────────────────────────────────────────────
# Resolves from Executables/ → CLIPDFQwen3VLChat/ → PythonApplications/ →
# repo root, then adds MoreMinerU into sys.path.
_APP_PATH = Path(__file__).resolve().parents[1]
_PROJECT_PATH = _APP_PATH.parents[1]

for _lib in [
    _PROJECT_PATH / "PythonLibraries" / "CoreCode",
    _PROJECT_PATH / "PythonLibraries" / "HuggingFace" / "MoreMinerU",
]:
    if _lib.exists() and str(_lib) not in sys.path:
        sys.path.append(str(_lib))


# ── Prompts ───────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = (
    "You are a precision P&ID (Piping and Instrumentation Diagram) parser. "
    "Extract only what is explicitly visible in the image. "
    "Output valid JSON only — no explanation, no markdown fences."
)

_NODE_PROMPT = """\
This is a P&ID (Piping and Instrumentation Diagram) schematic.

Identify every component visible in the diagram. For each component output a JSON object with:
  "tag"         — the label/number shown in the diagram (e.g. "N2", "7", "57", "9")
  "type"        — one of: vessel, regulator, pressure_gauge, solenoid_valve, check_valve, filter, orifice, igniter, sensor, manual_valve, relief_valve, flow_meter
  "description" — one short phrase: symbol shape + position in diagram

Output ONLY a JSON array of these objects. Example format:
[
  {"tag": "N2", "type": "vessel", "description": "N2 gas bottle, bottom center"},
  {"tag": "4",  "type": "check_valve", "description": "check valve with filter, below main line junction"}
]"""

_EDGE_PROMPT_TEMPLATE = """\
This is a P&ID (Piping and Instrumentation Diagram) schematic.

The components already identified in this diagram are:
{node_list}

Now identify all pipe/line connections between components. Follow the arrow directions — they indicate flow direction (source → destination).

For each connection output a JSON object with:
  "from"  — tag of the source component
  "to"    — tag of the destination component
  "label" — what flows (e.g. "N2 gas", "regulated N2", "vent to atm", "LOX pressurant")

Output ONLY a JSON array of edge objects. Example:
[
  {{"from": "N2", "to": "4",  "label": "N2 gas"}},
  {{"from": "4",  "to": "7",  "label": "N2 supply"}},
  {{"from": "7",  "to": "57", "label": "regulated N2"}}
]"""

_MERMAID_PROMPT_TEMPLATE = """\
Convert this P&ID topology JSON into a valid Mermaid flowchart.

Nodes:
{node_json}

Edges:
{edge_json}

Rules:
- Use `flowchart LR` (left to right)
- Node IDs: use the tag field (replace spaces/special chars with underscores)
- Node shape by type:
    vessel         → [(tag\\ntype)]
    regulator      → [tag\\nRegulator]
    pressure_gauge → ((tag\\nGauge))
    solenoid_valve → [tag\\nSolenoid]
    check_valve    → [tag\\nCheck Valve]
    filter         → [tag\\nFilter]
    manual_valve   → [tag\\nManual Valve]
    relief_valve   → [tag\\nRelief Valve]
    orifice        → [tag\\nOrifice]
    igniter        → [/tag\\nIgniter/]
    sensor         → ((tag\\nSensor))
    flow_meter     → ((tag\\nFlow Meter))
- Edge labels: use the label field in quotes
- Output ONLY the Mermaid code block (no backtick fences, no explanation)"""


# ── JSON parsing helpers ──────────────────────────────────────────────────────

def _try_parse_json(text: str) -> tuple[list | dict | None, str]:
    """Attempt to parse JSON from raw VLM text. Returns (parsed, error)."""
    stripped = text.strip()
    # strip markdown fences if the model added them despite instructions
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        stripped = "\n".join(
            line for line in lines
            if not line.strip().startswith("```")
        ).strip()
    try:
        return json.loads(stripped), ""
    except json.JSONDecodeError as exc:
        return None, str(exc)


def _nodes_to_list(nodes: list[dict]) -> str:
    return "\n".join(
        f"  - tag={n['tag']!r}, type={n.get('type', '?')!r}, "
        f"desc={n.get('description', '')!r}"
        for n in nodes
    )


# ── Mermaid synthesis (pure Python fallback) ──────────────────────────────────

_TYPE_TO_SHAPE = {
    "vessel":        "[({})]",
    "regulator":     "[{}]",
    "pressure_gauge": "(({})))",
    "solenoid_valve": "[{}]",
    "check_valve":   "[{}]",
    "filter":        "[{}]",
    "manual_valve":  "[{}]",
    "relief_valve":  "[{}]",
    "orifice":       "[{}]",
    "igniter":       "[/{}\\]",
    "sensor":        "(({})))",
    "flow_meter":    "(({})))",
}


def _safe_id(tag: str) -> str:
    return tag.replace(" ", "_").replace("-", "_").replace("/", "_")


def _build_mermaid(nodes: list[dict], edges: list[dict]) -> str:
    lines = ["flowchart LR"]
    for node in nodes:
        tag = node.get("tag", "?")
        node_type = node.get("type", "vessel")
        node_id = _safe_id(tag)
        label = f"{tag}\\n{node_type.replace('_', ' ').title()}"
        shape_tmpl = _TYPE_TO_SHAPE.get(node_type, "[{}]")
        lines.append(f"    {node_id}{shape_tmpl.format(label)}")
    lines.append("")
    for edge in edges:
        from_id = _safe_id(str(edge.get("from", "?")))
        to_id = _safe_id(str(edge.get("to", "?")))
        label = edge.get("label", "")
        if label:
            lines.append(f'    {from_id} -->|"{label}"| {to_id}')
        else:
            lines.append(f"    {from_id} --> {to_id}")
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Two-stage P&ID topology extraction via Qwen3-VL."
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path to the P&ID PNG/JPG image to parse.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Directory to write results into (created if absent).",
    )
    parser.add_argument(
        "--configpath",
        default=str(_APP_PATH),
        help=(
            "Directory containing Configurations/qwen3vl_configuration.yml. "
            f"Defaults to {_APP_PATH}."
        ),
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Max tokens per VLM call (default 2048).",
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    output_dir = Path(args.output)
    config_path = Path(args.configpath) / "Configurations" / "qwen3vl_configuration.yml"

    if not image_path.exists():
        print(f"ERROR: image not found: {image_path}")
        sys.exit(1)
    if not config_path.exists():
        print(f"ERROR: config not found: {config_path}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    report_lines: list[str] = []

    def log(msg: str) -> None:
        print(msg)
        report_lines.append(msg)

    log(f"P&ID Topology Extraction")
    log(f"  image:  {image_path}")
    log(f"  output: {output_dir}")
    log(f"  config: {config_path}")
    log("")

    # Load model
    from moremineru.Configurations.Qwen3VLConfiguration import Qwen3VLConfiguration
    from moremineru.Applications.Qwen3VLVLLM import Qwen3VLVLLM
    from PIL import Image

    config = Qwen3VLConfiguration.from_yaml(config_path)
    # Override system prompt for structured extraction mode
    config.system_prompt = _SYSTEM_PROMPT

    runner = Qwen3VLVLLM(config)
    log("Loading Qwen3-VL model...")
    t0 = time.time()
    runner.load()
    log(f"  loaded in {time.time() - t0:.1f}s")
    log("")

    image = Image.open(image_path).convert("RGB")
    # Deterministic sampling for structured extraction
    sampling = {"temperature": 0.0, "max_tokens": args.max_tokens}

    # ── Stage 1: Node extraction ──────────────────────────────────────────────
    log("Stage 1: Node extraction...")
    t1 = time.time()
    node_raw = runner.generate(image, _NODE_PROMPT, sampling_overrides=sampling)
    log(f"  completed in {time.time() - t1:.1f}s")
    log(f"  raw response ({len(node_raw)} chars):\n{node_raw}\n")

    (output_dir / "nodes_raw.txt").write_text(node_raw)
    nodes, node_err = _try_parse_json(node_raw)
    if nodes is None:
        log(f"  WARNING: failed to parse nodes JSON: {node_err}")
        nodes = []
    else:
        log(f"  parsed {len(nodes)} nodes")
    (output_dir / "nodes.json").write_text(json.dumps(nodes, indent=2))

    # ── Stage 2: Edge extraction ──────────────────────────────────────────────
    node_list_str = _nodes_to_list(nodes) if nodes else "  (no nodes parsed)"
    edge_prompt = _EDGE_PROMPT_TEMPLATE.format(node_list=node_list_str)

    log("Stage 2: Edge extraction...")
    t2 = time.time()
    edge_raw = runner.generate(image, edge_prompt, sampling_overrides=sampling)
    log(f"  completed in {time.time() - t2:.1f}s")
    log(f"  raw response ({len(edge_raw)} chars):\n{edge_raw}\n")

    (output_dir / "edges_raw.txt").write_text(edge_raw)
    edges, edge_err = _try_parse_json(edge_raw)
    if edges is None:
        log(f"  WARNING: failed to parse edges JSON: {edge_err}")
        edges = []
    else:
        log(f"  parsed {len(edges)} edges")
    (output_dir / "edges.json").write_text(json.dumps(edges, indent=2))

    # ── Stage 3: Mermaid synthesis ────────────────────────────────────────────
    log("Stage 3: Mermaid synthesis (Python)...")
    topology = {"nodes": nodes, "edges": edges}
    (output_dir / "topology.json").write_text(json.dumps(topology, indent=2))

    mermaid = _build_mermaid(nodes, edges)
    (output_dir / "topology.mermaid").write_text(mermaid)
    log(f"  Mermaid output:\n\n{mermaid}\n")

    (output_dir / "report.txt").write_text("\n".join(report_lines))
    log(f"Results written to {output_dir}")
    log(f"Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
