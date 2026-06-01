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
import type { CliImageConfig, CliImageStatus, LoraConfig } from "./types";

let config: CliImageConfig | null = null;

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
        <button class="tab active" data-tab="status">Status</button>
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
      <section class="panel active" data-panel="status">
        <div class="panel-title">
          <h2>Status</h2>
          <button id="copyStatus" title="Copy the Docker .status command to clipboard — paste it in a terminal running CLIImage">Copy Docker .status cmd</button>
        </div>
        <p class="muted" style="margin:0">Read-only snapshot of the current YAML values. Use the tabs to edit them.</p>
        <dl id="status" class="status-grid"></dl>
      </section>
      <section class="panel" data-panel="prompts">
        <div class="panel-title"><h2>Prompts</h2></div>
        <label>
          Prompt
          <span class="field-hint">→ <strong>CLIP</strong> encoder (clip-vit-large-patch14) · ≤77 tokens · sets overall composition &amp; style · longer text is silently truncated</span>
          <textarea id="prompt" rows="8"></textarea>
        </label>
        <label>
          Prompt 2
          <span class="field-hint">→ <strong>T5 XXL</strong> encoder (t5-v1_1-xxl) · up to 512 tokens · use for detailed descriptions &amp; scene specifics · falls back to Prompt if empty</span>
          <textarea id="prompt2" rows="4"></textarea>
        </label>
        <label>
          Negative Prompt
          <span class="field-hint">→ <strong>CLIP</strong> · what to push away from · <em>only active when True CFG Scale &gt; 1</em> (set on Generation tab) — ignored at default 1.0</span>
          <textarea id="negativePrompt" rows="5"></textarea>
        </label>
        <label>
          Negative Prompt 2
          <span class="field-hint">→ <strong>T5 XXL</strong> · richer "what to avoid" · <em>only active when True CFG Scale &gt; 1</em> · falls back to Negative Prompt if empty</span>
          <textarea id="negativePrompt2" rows="4"></textarea>
        </label>
      </section>
      <section class="panel" data-panel="generation">
        <div class="panel-title"><h2>Generation</h2></div>
        <div class="field-grid">
          <label>Width<input id="width" type="number" min="64" step="8"></label>
          <label>Height<input id="height" type="number" min="64" step="8"></label>
          <label>
            Num. Inference Steps
            <span class="field-hint">Denoising iterations the scheduler runs. More = sharper detail, slower generation. Typical: 20–30. Below 10 may be too noisy; above 50 rarely improves results.</span>
            <input id="steps" type="number" min="1" step="1">
          </label>
          <label>
            Guidance Scale
            <span class="field-hint">Embedded conditioning that steers output toward the prompt — a signal baked into Flux's distilled architecture (no extra model pass). Low (1–2) = loose, creative; typical (2.5–4.5) = balanced; high (5+) = very literal, may over-constrain. Default: 3.5.</span>
            <input id="guidance" type="number" step="0.05">
          </label>
          <label>
            True CFG Scale
            <span class="field-hint"><strong>CFG = Classifier-Free Guidance.</strong> When &gt; 1 and a Negative Prompt is set, runs <em>two</em> denoiser passes per step — positive + unconditioned — then blends: output = uncond + scale × (cond − uncond). <strong>Doubles inference time.</strong> 1.0 = off (default); 1.1–2.0 = subtle negative push; 2.5–4.0+ = strong negative avoidance.</span>
            <input id="trueCfg" type="number" step="0.05">
          </label>
          <label>
            Seed
            <span class="field-hint">Pins the random noise initializer. Same seed + same settings = same image. Clear to generate a fresh random seed each run.</span>
            <div class="seed-row">
              <input id="seed" type="number" step="1" min="0" placeholder="random">
              <button id="randomSeed" type="button">Random</button>
            </div>
          </label>
        </div>
        <label>Output Path<input id="outputPath" type="text"></label>
        <h3 class="section-heading">Batch Processing</h3>
        <p class="profile-hint">
          <strong>.process_batch</strong> generates <em>N</em> images in one run, sweeping Guidance Scale
          from its starting value above and adding the Guidance Step after each image — a
          quick way to compare the same prompt across a range of guidance strengths in one shot.<br>
          Example: Guidance = 2.4, Step = 2.12, Images = 3 → image 0 at 2.4 · image 1 at 4.52 · image 2 at 6.64.
        </p>
        <div class="field-grid">
          <label>
            Number of Images
            <span class="field-hint">Total images to generate per batch run (each at a different guidance level).</span>
            <input id="batchImages" type="number" min="1" step="1">
          </label>
          <label>
            Guidance Step
            <span class="field-hint">Amount added to Guidance Scale after each image. Set to 0 to keep guidance constant across all batch images.</span>
            <input id="guidanceStep" type="number" step="0.01">
          </label>
          <label>
            Base Filename
            <span class="field-hint">Prefix for saved files. Full name: {base}{model}-Steps{n}Iter{i}-Guidance{g}cfg{c}-{hash}.png</span>
            <input id="baseFilename" type="text">
          </label>
        </div>
      </section>
      <section class="panel" data-panel="loras">
        <div class="panel-title">
          <h2>LoRAs</h2>
          <input id="loraFilter" type="search" placeholder="Filter LoRAs">
        </div>
        <label class="field-narrow">
          Global LoRA Scale
          <span class="field-hint">Master multiplier passed to the pipeline's attention mechanism for all active LoRAs — on top of each LoRA's individual strength. <strong>Leave blank to omit this key entirely from the YAML</strong> (Python reads absence as None, which means no global override).</span>
          <input id="loraScale" type="number" step="0.05" min="0" placeholder="blank = key omitted from YAML">
        </label>
        <div id="loraList" class="lora-list"></div>
      </section>
      <section class="panel" data-panel="profiles">
        <div class="panel-title"><h2>Profiles</h2></div>
        <p class="muted profile-hint">
          A profile is a named snapshot of all six YAML files saved together
          as a subdirectory inside the active config directory.<br>
          <strong>Save</strong> copies the current live files into
          <code id="profilesDir">…/profiles/&lt;name&gt;/</code>.<br>
          <strong>Apply</strong> copies them back, automatically backing up the
          live files to <code>profiles/_backups/&lt;timestamp&gt;/</code> first.
        </p>
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
  setInput("#guidance", config.flux.guidance_scale);
  setInput("#trueCfg", config.flux.true_cfg_scale);
  setInput("#seed", config.flux.seed);
  setInput("#outputPath", config.flux.temporary_save_path);
  setInput("#batchImages", config.batch.number_of_images);
  setInput("#guidanceStep", config.batch.guidance_scale_step);
  setInput("#baseFilename", config.batch.base_filename);
  setInput("#loraScale", config.loras.lora_scale);
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
  if (lora.description) {
    const desc = document.createElement("div");
    desc.className = "muted";
    desc.textContent = lora.description;
    main.append(desc);
  }

  if (lora.description) {
    const desc = document.createElement("div");
    desc.className = "lora-description";
    desc.textContent = lora.description;
    main.append(desc);
  }

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
  const seed = readNumber("#seed");
  if (seed !== null) {
    config.flux.seed = seed;
  } else {
    delete (config.flux as Record<string, unknown>).seed;
  }
  config.batch.number_of_images = readNumber("#batchImages") ?? undefined;
  config.batch.guidance_scale_step = readNumber("#guidanceStep") ?? undefined;
  config.batch.base_filename = getElement<HTMLInputElement>("#baseFilename").value;
  // lora_scale: if blank, delete the key entirely so it is absent from the YAML.
  // Python's from_yaml uses `if "lora_scale" in data` — a null/~ value still
  // counts as "present", so the key must not appear at all to mean "no global scale".
  const loraScale = readNumber("#loraScale");
  if (loraScale !== null) {
    config.loras.lora_scale = loraScale;
  } else {
    delete (config.loras as Record<string, unknown>).lora_scale;
  }
  // Preserve required flux fields that are not exposed as form controls.
  // FluxGenerationConfiguration.from_yaml() validates that every field
  // NOT in FIELDS_TO_EXCLUDE = {"seed", "temporary_save_path"} must be
  // present as a key in the YAML.  If a prior bad write stripped one of
  // these, default it to null so JSON.stringify keeps the key in the
  // payload and serde_yaml writes it back as `~` (null).
  const fluxRecord = config.flux as Record<string, unknown>;
  for (const field of ["num_images_per_prompt", "max_sequence_length"] as const) {
    if (!(field in fluxRecord)) {
      fluxRecord[field] = null;
    }
  }
  return config;
}

async function reload() {
  config = await loadConfig();
  const status = await loadStatus();
  const location = await loadLocation();
  renderForm();
  renderStatus(status);
  setInput("#configDir", location.config_dir);
  // Show the concrete profiles directory so users know exactly where saves land
  const profilesDirEl = document.querySelector<HTMLElement>("#profilesDir");
  if (profilesDirEl) {
    profilesDirEl.textContent = `${location.config_dir}/profiles/<name>/`;
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

getElement<HTMLButtonElement>("#reload").addEventListener("click", () =>
  reload().catch((error) => {
    getElement<HTMLParagraphElement>("#state").textContent = "Reload failed";
    showToast(error.message);
    console.error("reload error:", error);
  })
);
getElement<HTMLButtonElement>("#save").addEventListener("click", () =>
  saveYaml().catch((error) => {
    showToast(`Save failed: ${error.message}`);
    console.error("save error:", error);
  })
);
getElement<HTMLInputElement>("#loraFilter").addEventListener("input", () => renderLoras());
getElement<HTMLButtonElement>("#randomSeed").addEventListener("click", () => {
  // 32-bit unsigned integer — enough entropy and compatible with all seed consumers
  const seed = Math.floor(Math.random() * 4_294_967_296);
  setInput("#seed", seed);
});
getElement<HTMLButtonElement>("#saveProfile").addEventListener("click", () => {
  const name = getElement<HTMLInputElement>("#profileName").value.trim();
  if (!name) {
    showToast("Enter a profile name");
    return;
  }
  saveProfile(name)
    .then(() => reload())
    .then(() => showToast(`Saved ${name}`))
    .catch((error) => {
      showToast(`Save profile failed: ${error.message}`);
      console.error("saveProfile error:", error);
    });
});
getElement<HTMLButtonElement>("#useDefaultDir").addEventListener("click", () =>
  loadLocation()
    .then((location) => setInput("#configDir", location.default_config_dir))
    .catch((error) => {
      showToast(`Load failed: ${error.message}`);
      console.error("useDefaultDir error:", error);
    })
);
getElement<HTMLButtonElement>("#applyConfigDir").addEventListener("click", () =>
  setLocation(
    getElement<HTMLInputElement>("#configDir").value.trim(),
    getElement<HTMLInputElement>("#initializeFromExamples").checked
  )
    .then(() => reload())
    .then(() => showToast("Config directory applied"))
    .catch((error) => {
      showToast(`Apply failed: ${error.message}`);
      console.error("applyConfigDir error:", error);
    })
);
getElement<HTMLButtonElement>("#copyStatus").addEventListener("click", async () => {
  await navigator.clipboard.writeText(
    "python3 Scripts/QuickAliases/RunCLIImageNunchaku.py --command .status"
  );
  showToast("Copied command");
});

reload().catch((error) => {
  getElement<HTMLParagraphElement>("#state").textContent = "Backend unavailable";
  showToast(error.message);
  console.error("startup reload failed:", error);
});
