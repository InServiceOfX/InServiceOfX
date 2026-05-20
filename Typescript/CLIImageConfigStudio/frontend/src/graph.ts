import { LGraph, LGraphCanvas, LiteGraph } from "litegraph.js";
import "litegraph.js/css/litegraph.css";
import type { CliImageConfig } from "./types";

export function createConfigGraph(canvas: HTMLCanvasElement): LGraph {
  const graph = new LGraph();
  const graphCanvas = new LGraphCanvas(canvas, graph);
  graphCanvas.background_image = "";
  graphCanvas.ds.scale = 0.85;

  const nodes = [
    ["CLIImage/profile", "Profile", [20, 40]],
    ["CLIImage/models", "Models", [250, 40]],
    ["CLIImage/prompts", "Prompts", [480, 40]],
    ["CLIImage/loras", "LoRAs", [250, 210]],
    ["CLIImage/generation", "Generation", [480, 210]],
    ["CLIImage/output", "Output", [720, 130]]
  ] as const;

  for (const [type, title, position] of nodes) {
    const node = LiteGraph.createNode(type);
    node.title = title;
    node.pos = [...position];
    graph.add(node);
  }

  graph.start();
  return graph;
}

export function registerNodes() {
  class ProfileNode {
    title = "Profile";
    addOutput!: (name: string, type: string) => void;
    properties!: Record<string, unknown>;
    constructor() {
      this.addOutput("yaml", "profile");
      this.properties = { active: "" };
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
      this.properties = { size: "" };
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
  }

  LiteGraph.registerNodeType("CLIImage/profile", ProfileNode as never);
  LiteGraph.registerNodeType("CLIImage/models", ModelNode as never);
  LiteGraph.registerNodeType("CLIImage/prompts", PromptNode as never);
  LiteGraph.registerNodeType("CLIImage/loras", LoraNode as never);
  LiteGraph.registerNodeType("CLIImage/generation", GenerationNode as never);
  LiteGraph.registerNodeType("CLIImage/output", OutputNode as never);
}

export function updateGraphProperties(graph: LGraph, config: CliImageConfig) {
  const loras = config.loras.loras ?? [];
  const activeLoras = loras.filter((lora) => lora.is_active);
  for (const node of (graph as unknown as { _nodes: any[] })._nodes) {
    if (node.type === "CLIImage/models") {
      const paths = config.nunchaku.nunchaku_model_paths;
      node.properties.count = Array.isArray(paths) ? paths.length : paths ? 1 : 0;
    }
    if (node.type === "CLIImage/prompts") {
      node.properties.preview = (config.pipeline.prompt ?? "").slice(0, 80);
    }
    if (node.type === "CLIImage/loras") {
      node.properties.active = activeLoras.length;
    }
    if (node.type === "CLIImage/generation") {
      node.properties.size = `${config.flux.width ?? ""}x${config.flux.height ?? ""}`;
    }
    if (node.type === "CLIImage/output") {
      node.properties.path = config.flux.temporary_save_path ?? "";
    }
  }
  graph.setDirtyCanvas(true, true);
}
