import "./styles.css";
import {
  applyProfile,
  loadConfig,
  loadLocation,
  loadStatus,
  saveConfig,
  saveProfile,
  setLocation
} from "./api";
import { createConfigGraph, registerNodes, updateGraphProperties } from "./graph";
import type { CliImageConfig, CliImageStatus, LoraConfig } from "./types";

let config: CliImageConfig | null = null;
let graph = null as ReturnType<typeof createConfigGraph> | null;

const app = document.querySelector<HTMLDivElement>("#app");
if (!app) {
  throw new Error("#app not found");
}

app.innerHTML = `
  <main class="app">
    <aside class="sidebar">
      <header>
        <h1>CLIImage Config Studio</h1>
        <p id="state">Loading live YAML</p>
      </header>
      <nav class="tabs">
        <button class="tab active" data-tab="overview">Overview</button>
        <button class="tab" data-tab="prompts">Prompts</button>
        <button class="tab" data-tab="generation">Generation</button>
        <button class="tab" data-tab="loras">LoRAs</button>
        <button class="tab" data-tab="profiles">Profiles</button>
        <button class="tab" data-tab="location">Location</button>
      </nav>
      <div class="sidebar-actions">
        <button id="reload">Reload YAML</button>
        <button id="save" class="primary">Save YAML</button>
      </div>
    </aside>
    <section class="workspace">
      <section class="panel active" data-panel="overview">
        <div class="panel-title">
          <h2>Workflow</h2>
          <button id="copyStatus">Copy Status Command</button>
        </div>
        <canvas id="graphCanvas" width="980" height="360"></canvas>
        <dl id="status" class="status-grid"></dl>
      </section>
      <section class="panel" data-panel="prompts">
        <div class="panel-title"><h2>Prompts</h2></div>
        <label>Prompt<textarea id="prompt" rows="8"></textarea></label>
        <label>Prompt 2<textarea id="prompt2" rows="4"></textarea></label>
        <label>Negative Prompt<textarea id="negativePrompt" rows="5"></textarea></label>
        <label>Negative Prompt 2<textarea id="negativePrompt2" rows="4"></textarea></label>
      </section>
      <section class="panel" data-panel="generation">
        <div class="panel-title"><h2>Generation</h2></div>
        <div class="field-grid">
          <label>Width<input id="width" type="number" min="64" step="8"></label>
          <label>Height<input id="height" type="number" min="64" step="8"></label>
          <label>Steps<input id="steps" type="number" min="1" step="1"></label>
          <label>Images<input id="images" type="number" min="1" step="1"></label>
          <label>Guidance<input id="guidance" type="number" step="0.05"></label>
          <label>True CFG<input id="trueCfg" type="number" step="0.05"></label>
        </div>
        <label>Output Path<input id="outputPath" type="text"></label>
      </section>
      <section class="panel" data-panel="loras">
        <div class="panel-title">
          <h2>LoRAs</h2>
          <input id="loraFilter" type="search" placeholder="Filter LoRAs">
        </div>
        <div id="loraList" class="lora-list"></div>
      </section>
      <section class="panel" data-panel="profiles">
        <div class="panel-title"><h2>Profiles</h2></div>
        <div class="profile-tools">
          <input id="profileName" type="text" placeholder="profile-name">
          <button id="saveProfile">Save Profile</button>
        </div>
        <div id="profiles" class="profile-list"></div>
      </section>
      <section class="panel" data-panel="location">
        <div class="panel-title"><h2>Config Directory</h2></div>
        <label>Active Directory<input id="configDir" type="text"></label>
        <label class="toggle">
          <input id="initializeFromExamples" type="checkbox">
          Initialize missing YAML from .example templates
        </label>
        <div class="button-row">
          <button id="useDefaultDir">Use Default</button>
          <button id="applyConfigDir" class="primary">Apply Directory</button>
        </div>
        <p class="muted">
          Absolute paths are accepted. Relative paths resolve from the
          InServiceOfX repo root.
        </p>
      </section>
    </section>
  </main>
  <div id="toast"></div>
`;

function getElement<T extends HTMLElement>(selector: string): T {
  const element = document.querySelector<T>(selector);
  if (!element) {
    throw new Error(`${selector} not found`);
  }
  return element;
}

function setInput(selector: string, value: unknown) {
  getElement<HTMLInputElement | HTMLTextAreaElement>(selector).value =
    value === undefined || value === null ? "" : String(value);
}

function readNumber(selector: string): number | null {
  const value = getElement<HTMLInputElement>(selector).value.trim();
  return value === "" ? null : Number(value);
}

function showToast(message: string) {
  const toast = getElement<HTMLDivElement>("#toast");
  toast.textContent = message;
  toast.classList.add("visible");
  window.setTimeout(() => toast.classList.remove("visible"), 2400);
}

function renderStatus(status: CliImageStatus) {
  const rows: [string, unknown][] = [
    ["CUDA", status.cuda_device],
    ["FLUX model", status.flux_model_path],
    ["Nunchaku models", status.nunchaku_model_count],
    ["Dimensions", `${status.width ?? ""} x ${status.height ?? ""}`],
    ["Steps", status.steps],
    ["Guidance", status.guidance_scale],
    ["True CFG", status.true_cfg_scale],
    ["Output path", status.output_path],
    ["Batch images", status.batch_images],
    ["Prompt", status.prompt],
    ["Negative", status.negative_prompt],
    ["Active LoRAs", status.active_lora_count],
    [
      "LoRA names",
      (status.active_loras ?? [])
        .map((lora) => `${lora.nickname} (${lora.lora_strength})`)
        .join(", ")
    ]
  ];
  const statusGrid = getElement<HTMLDListElement>("#status");
  statusGrid.textContent = "";
  for (const [label, value] of rows) {
    const dt = document.createElement("dt");
    dt.textContent = label;
    const dd = document.createElement("dd");
    dd.textContent = value === undefined || value === null ? "" : String(value);
    statusGrid.append(dt, dd);
  }
}

function renderForm() {
  if (!config) return;
  setInput("#prompt", config.pipeline.prompt);
  setInput("#prompt2", config.pipeline.prompt_2);
  setInput("#negativePrompt", config.pipeline.negative_prompt);
  setInput("#negativePrompt2", config.pipeline.negative_prompt_2);
  setInput("#width", config.flux.width);
  setInput("#height", config.flux.height);
  setInput("#steps", config.flux.num_inference_steps);
  setInput("#images", config.batch.number_of_images);
  setInput("#guidance", config.flux.guidance_scale);
  setInput("#trueCfg", config.flux.true_cfg_scale);
  setInput("#outputPath", config.flux.temporary_save_path);
  renderLoras();
  renderProfiles();
}

function renderLoras() {
  if (!config) return;
  const filter = getElement<HTMLInputElement>("#loraFilter").value.toLowerCase();
  const list = getElement<HTMLDivElement>("#loraList");
  list.textContent = "";
  for (const lora of config.loras.loras ?? []) {
    const searchable = [
      lora.nickname,
      lora.filename,
      lora.directory_path,
      lora.description
    ].join(" ").toLowerCase();
    if (filter && !searchable.includes(filter)) continue;
    list.append(createLoraRow(lora));
  }
}

function createLoraRow(lora: LoraConfig): HTMLElement {
  const row = document.createElement("div");
  row.className = "lora-row";
  const main = document.createElement("div");
  const name = document.createElement("div");
  name.className = "strong";
  name.textContent = lora.nickname;
  const path = document.createElement("div");
  path.className = "muted path";
  path.textContent = `${lora.directory_path}/${lora.filename}`;
  main.append(name, path);

  const active = document.createElement("label");
  active.className = "toggle";
  const checkbox = document.createElement("input");
  checkbox.type = "checkbox";
  checkbox.checked = Boolean(lora.is_active);
  checkbox.addEventListener("change", () => {
    lora.is_active = checkbox.checked;
  });
  active.append(checkbox, document.createTextNode("Active"));

  const strength = document.createElement("input");
  strength.type = "number";
  strength.step = "0.05";
  strength.value = String(lora.lora_strength ?? 1);
  strength.addEventListener("input", () => {
    lora.lora_strength = Number(strength.value);
  });

  row.append(main, active, strength);
  return row;
}

function renderProfiles() {
  if (!config) return;
  const list = getElement<HTMLDivElement>("#profiles");
  list.textContent = "";
  for (const profile of config.profiles) {
    const row = document.createElement("div");
    row.className = "profile-row";
    const name = document.createElement("div");
    name.textContent = profile.name;
    const state = document.createElement("span");
    state.className = profile.complete ? "badge" : "badge warn";
    state.textContent = profile.complete ? "complete" : "incomplete";
    const apply = document.createElement("button");
    apply.textContent = "Apply";
    apply.addEventListener("click", async () => {
      await applyProfile(profile.name);
      await reload();
      showToast(`Applied ${profile.name}`);
    });
    row.append(name, state, apply);
    list.append(row);
  }
}

function updateConfigFromForm(): CliImageConfig {
  if (!config) throw new Error("config not loaded");
  config.pipeline.prompt = getElement<HTMLTextAreaElement>("#prompt").value;
  config.pipeline.prompt_2 = getElement<HTMLTextAreaElement>("#prompt2").value;
  config.pipeline.negative_prompt = getElement<HTMLTextAreaElement>("#negativePrompt").value;
  config.pipeline.negative_prompt_2 = getElement<HTMLTextAreaElement>("#negativePrompt2").value;
  config.flux.width = readNumber("#width") ?? undefined;
  config.flux.height = readNumber("#height") ?? undefined;
  config.flux.num_inference_steps = readNumber("#steps") ?? undefined;
  config.flux.guidance_scale = readNumber("#guidance") ?? undefined;
  config.flux.true_cfg_scale = readNumber("#trueCfg") ?? undefined;
  config.flux.temporary_save_path = getElement<HTMLInputElement>("#outputPath").value;
  config.batch.number_of_images = readNumber("#images") ?? undefined;
  return config;
}

async function reload() {
  config = await loadConfig();
  const status = await loadStatus();
  const location = await loadLocation();
  renderForm();
  renderStatus(status);
  setInput("#configDir", location.config_dir);
  if (graph) {
    updateGraphProperties(graph, config);
  }
  getElement<HTMLParagraphElement>("#state").textContent = "Editing live YAML";
}

async function saveYaml() {
  const updated = updateConfigFromForm();
  await saveConfig({
    batch: updated.batch,
    flux: updated.flux,
    loras: updated.loras,
    pipeline: updated.pipeline
  });
  await reload();
  showToast("Saved YAML");
}

function setupTabs() {
  for (const tab of document.querySelectorAll<HTMLButtonElement>(".tab")) {
    tab.addEventListener("click", () => {
      const tabName = tab.dataset.tab;
      for (const candidate of document.querySelectorAll(".tab")) {
        candidate.classList.toggle("active", candidate === tab);
      }
      for (const panel of document.querySelectorAll<HTMLElement>(".panel")) {
        panel.classList.toggle("active", panel.dataset.panel === tabName);
      }
    });
  }
}

setupTabs();
registerNodes();
graph = createConfigGraph(getElement<HTMLCanvasElement>("#graphCanvas"));
getElement<HTMLButtonElement>("#reload").addEventListener("click", () => reload());
getElement<HTMLButtonElement>("#save").addEventListener("click", () => saveYaml());
getElement<HTMLInputElement>("#loraFilter").addEventListener("input", () => renderLoras());
getElement<HTMLButtonElement>("#saveProfile").addEventListener("click", async () => {
  const name = getElement<HTMLInputElement>("#profileName").value.trim();
  if (!name) {
    showToast("Enter a profile name");
    return;
  }
  await saveProfile(name);
  await reload();
  showToast(`Saved ${name}`);
});
getElement<HTMLButtonElement>("#useDefaultDir").addEventListener("click", async () => {
  const location = await loadLocation();
  setInput("#configDir", location.default_config_dir);
});
getElement<HTMLButtonElement>("#applyConfigDir").addEventListener("click", async () => {
  const configDir = getElement<HTMLInputElement>("#configDir").value.trim();
  const initialize = getElement<HTMLInputElement>("#initializeFromExamples").checked;
  await setLocation(configDir, initialize);
  await reload();
  showToast("Config directory applied");
});
getElement<HTMLButtonElement>("#copyStatus").addEventListener("click", async () => {
  await navigator.clipboard.writeText(
    "python3 Scripts/QuickAliases/RunCLIImageNunchaku.py --command .status"
  );
  showToast("Copied command");
});

reload().catch((error) => {
  getElement<HTMLParagraphElement>("#state").textContent = "Backend unavailable";
  showToast(error.message);
});
