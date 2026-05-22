import { LGraph, LGraphCanvas, LiteGraph } from "litegraph.js";
import "litegraph.js/css/litegraph.css";
import type { CliImageConfig } from "./types";

// ---------------------------------------------------------------------------
// Wire colours per slot type
// ---------------------------------------------------------------------------
const WIRE_COLORS: Record<string, string> = {
  profile: "#4a9eff",
  model: "#7ab8ff",
  prompt: "#4aff8a",
  loras: "#ffaa4a",
  image: "#bf80ff",
};

// ---------------------------------------------------------------------------
// Per-node-type colours (title bar / body)
// ---------------------------------------------------------------------------
const NODE_THEME: Record<string, { color: string; bgcolor: string }> = {
  "CLIImage/profile":    { color: "#1b3a5c", bgcolor: "#0f2035" },
  "CLIImage/models":     { color: "#0f3545", bgcolor: "#08202a" },
  "CLIImage/prompts":    { color: "#1a4a2a", bgcolor: "#0f2a16" },
  "CLIImage/loras":      { color: "#4a3010", bgcolor: "#2a1a08" },
  "CLIImage/generation": { color: "#2a1a4a", bgcolor: "#170f2a" },
  "CLIImage/output":     { color: "#2e2e2e", bgcolor: "#1a1a1a" },
};

// ---------------------------------------------------------------------------
// Drawing helpers
// ---------------------------------------------------------------------------
function truncate(text: string, max: number): string {
  return text.length > max ? text.slice(0, max - 1) + "…" : text;
}

/** Draw a small label line followed by a value line inside a node. */
function drawRow(
  ctx: CanvasRenderingContext2D,
  label: string,
  value: string,
  y: number,
  nodeWidth: number
) {
  const maxChars = Math.max(8, Math.floor((nodeWidth - 16) / 6.5));
  ctx.fillStyle = "#6a8399";
  ctx.font = "10px sans-serif";
  ctx.fillText(label.toUpperCase(), 8, y);
  ctx.fillStyle = "#dce8f4";
  ctx.font = "bold 11px sans-serif";
  ctx.fillText(truncate(value || "—", maxChars), 8, y + 14);
}

// ---------------------------------------------------------------------------
// Node class definitions
// ---------------------------------------------------------------------------

class ProfileNode {
  title = "Profile";
  addOutput!: (name: string, type: string) => void;
  properties!: Record<string, unknown>;
  constructor() {
    this.addOutput("yaml", "profile");
    this.properties = { active: "" };
  }
  onDrawForeground(ctx: CanvasRenderingContext2D) {
    if ((this as any).flags?.collapsed) return;
    drawRow(ctx, "active profile", String(this.properties.active || "—"), 16, (this as any).size?.[0] ?? 150);
  }
}

class ModelNode {
  title = "Models";
  addInput!: (name: string, type: string) => void;
  addOutput!: (name: string, type: string) => void;
  properties!: Record<string, unknown>;
  constructor() {
    this.addInput("profile", "profile");
    this.addOutput("model", "model");
    this.properties = { count: 0 };
  }
  onDrawForeground(ctx: CanvasRenderingContext2D) {
    if ((this as any).flags?.collapsed) return;
    drawRow(ctx, "nunchaku models", String(this.properties.count ?? 0), 16, (this as any).size?.[0] ?? 150);
  }
}

class PromptNode {
  title = "Prompts";
  addInput!: (name: string, type: string) => void;
  addOutput!: (name: string, type: string) => void;
  properties!: Record<string, unknown>;
  constructor() {
    this.addInput("profile", "profile");
    this.addOutput("prompt", "prompt");
    this.properties = { preview: "" };
  }
  onDrawForeground(ctx: CanvasRenderingContext2D) {
    if ((this as any).flags?.collapsed) return;
    drawRow(ctx, "prompt", String(this.properties.preview || "—"), 16, (this as any).size?.[0] ?? 190);
  }
}

class LoraNode {
  title = "LoRAs";
  addInput!: (name: string, type: string) => void;
  addOutput!: (name: string, type: string) => void;
  properties!: Record<string, unknown>;
  constructor() {
    this.addInput("profile", "profile");
    this.addOutput("loras", "loras");
    this.properties = { active: 0 };
  }
  onDrawForeground(ctx: CanvasRenderingContext2D) {
    if ((this as any).flags?.collapsed) return;
    drawRow(ctx, "active LoRAs", String(this.properties.active ?? 0), 16, (this as any).size?.[0] ?? 150);
  }
}

class GenerationNode {
  title = "Generation";
  addInput!: (name: string, type: string) => void;
  addOutput!: (name: string, type: string) => void;
  properties!: Record<string, unknown>;
  constructor() {
    this.addInput("prompt", "prompt");
    this.addInput("loras", "loras");
    this.addOutput("image", "image");
    this.properties = { size: "", steps: "" };
  }
  onDrawForeground(ctx: CanvasRenderingContext2D) {
    if ((this as any).flags?.collapsed) return;
    const w = (this as any).size?.[0] ?? 190;
    drawRow(ctx, "dimensions", String(this.properties.size || "—"), 16, w);
    drawRow(ctx, "steps", String(this.properties.steps || "—"), 48, w);
  }
}

class OutputNode {
  title = "Output";
  addInput!: (name: string, type: string) => void;
  properties!: Record<string, unknown>;
  constructor() {
    this.addInput("image", "image");
    this.properties = { path: "" };
  }
  onDrawForeground(ctx: CanvasRenderingContext2D) {
    if ((this as any).flags?.collapsed) return;
    drawRow(ctx, "save path", String(this.properties.path || "—"), 16, (this as any).size?.[0] ?? 180);
  }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

export function registerNodes() {
  // Wire colours
  const lg = LiteGraph as unknown as Record<string, unknown>;
  if (typeof lg["slot_types_default_color"] === "object" && lg["slot_types_default_color"]) {
    const stc = lg["slot_types_default_color"] as Record<string, string>;
    Object.assign(stc, WIRE_COLORS);
  }

  LiteGraph.registerNodeType("CLIImage/profile",    ProfileNode as never);
  LiteGraph.registerNodeType("CLIImage/models",     ModelNode as never);
  LiteGraph.registerNodeType("CLIImage/prompts",    PromptNode as never);
  LiteGraph.registerNodeType("CLIImage/loras",      LoraNode as never);
  LiteGraph.registerNodeType("CLIImage/generation", GenerationNode as never);
  LiteGraph.registerNodeType("CLIImage/output",     OutputNode as never);
}

/** Node layout: [type, display title, [x, y], [w, h], tab to switch to on dbl-click] */
const NODE_LAYOUT = [
  ["CLIImage/profile",    "Profile",    [20,  50],  [155, 68], "profiles"],
  ["CLIImage/models",     "Models",     [250, 50],  [160, 68], "location"],
  ["CLIImage/prompts",    "Prompts",    [490, 50],  [190, 68], "prompts"],
  ["CLIImage/loras",      "LoRAs",      [250, 220], [160, 68], "loras"],
  ["CLIImage/generation", "Generation", [490, 220], [190, 88], "generation"],
  ["CLIImage/output",     "Output",     [740, 135], [185, 68], "generation"],
] as const;

export function createConfigGraph(
  canvas: HTMLCanvasElement,
  switchTab: (tab: string) => void
): LGraph {
  const graph = new LGraph();
  const graphCanvas = new LGraphCanvas(canvas, graph);
  graphCanvas.background_image = "";
  graphCanvas.ds.scale = 0.82;
  graphCanvas.ds.offset = [0, 0];

  const created: ReturnType<typeof LiteGraph.createNode>[] = [];

  for (const [type, title, pos, size, tab] of NODE_LAYOUT) {
    const node = LiteGraph.createNode(type);
    node.title = title as string;
    node.pos = [...pos] as [number, number];
    node.size = [...size] as [number, number];

    const theme = NODE_THEME[type as string];
    if (theme) {
      node.color   = theme.color;
      node.bgcolor = theme.bgcolor;
    }

    // Double-click → jump to the relevant sidebar tab
    (node as unknown as Record<string, unknown>)["onDblClick"] = () => switchTab(tab as string);

    graph.add(node);
    created.push(node);
  }

  // Wire connections (all nodes are in graph now)
  const [profile, models, prompts, loras, generation, output] = created;

  profile.connect(0, models,      0); // yaml  → models[profile]
  profile.connect(0, prompts,     0); // yaml  → prompts[profile]
  profile.connect(0, loras,       0); // yaml  → loras[profile]
  prompts.connect(0, generation,  0); // prompt → generation[prompt]
  loras.connect(  0, generation,  1); // loras  → generation[loras]
  generation.connect(0, output,   0); // image  → output[image]

  graph.start();
  return graph;
}

export function updateGraphProperties(graph: LGraph, config: CliImageConfig) {
  const loraList = config.loras.loras ?? [];
  const activeLoras = loraList.filter((l) => l.is_active);

  for (const node of (graph as unknown as { _nodes: any[] })._nodes) {
    switch (node.type) {
      case "CLIImage/models": {
        const paths = (config.nunchaku as Record<string, unknown>)["nunchaku_model_paths"];
        node.properties.count = Array.isArray(paths) ? paths.length : paths ? 1 : 0;
        break;
      }
      case "CLIImage/prompts":
        node.properties.preview = String(config.pipeline.prompt ?? "").slice(0, 55);
        break;
      case "CLIImage/loras":
        node.properties.active = activeLoras.length;
        break;
      case "CLIImage/generation":
        node.properties.size  = `${config.flux.width ?? "?"}×${config.flux.height ?? "?"}`;
        node.properties.steps = config.flux.num_inference_steps ?? "—";
        break;
      case "CLIImage/output":
        node.properties.path = config.flux.temporary_save_path ?? "";
        break;
    }
  }

  graph.setDirtyCanvas(true, true);
}
