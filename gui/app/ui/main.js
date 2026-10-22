// Frontend for the Tauri shell. All choices come from the Rust provider
// hooks, so this file knows nothing about any specific provider or model.

const invoke = window.__TAURI__?.core?.invoke ?? mockInvoke;

const $ = (id) => document.getElementById(id);

const state = {
  providers: [],
  provider: null,
  model: null,
  resolution: null,
  quality: null,
  busy: false,
  results: [],
  current: -1,
};

function renderChips(container, options, selected, onPick) {
  container.replaceChildren(
    ...options.map((opt) => {
      const chip = document.createElement("button");
      chip.type = "button";
      chip.className = "chip";
      chip.role = "radio";
      chip.textContent = opt;
      chip.setAttribute("aria-checked", String(opt === selected));
      chip.addEventListener("click", () => opt !== selected && onPick(opt));
      return chip;
    }),
  );
}

function currentProvider() {
  return state.providers.find((p) => p.name === state.provider);
}

function currentModel() {
  return currentProvider()?.models.find((m) => m.name === state.model);
}

async function selectProvider(name) {
  state.provider = name;
  const info = currentProvider();
  renderChips($("providers"), state.providers.map((p) => p.name), name, selectProvider);
  const note = $("status-note");
  note.textContent = info.status_note ?? "";
  note.hidden = !info.status_note;
  note.classList.toggle("ok", /^Authenticated/.test(info.status_note ?? ""));
  await selectModel(info.default_model);
}

async function selectModel(name) {
  state.model = name;
  renderChips($("models"), currentProvider().models.map((m) => m.name), name, selectModel);
  renderDropzone();
  const choices = await invoke("resolution_choices", { providerName: state.provider, model: name });
  await selectResolution(choices.default, choices.options);
}

async function selectResolution(value, options) {
  state.resolution = value;
  const render = (opts) => renderChips($("resolutions"), opts, value, (v) => selectResolution(v, opts));
  render(options);
  const choices = await invoke("quality_choices", {
    providerName: state.provider,
    model: state.model,
    resolution: value,
  });
  await selectQuality(choices.default, choices.options);
}

async function selectQuality(value, options) {
  state.quality = value;
  renderChips($("qualities"), options, value, (v) => selectQuality(v, options));
  await refreshEstimate();
}

function renderDropzone() {
  const max = currentModel()?.max_input_images;
  const zone = $("dropzone");
  zone.classList.toggle("disabled", max === 0);
  zone.textContent =
    max === 0
      ? "This model is text-to-image only: it takes no reference images."
      : `Accepts ${max ?? "any number of"} reference image${max === 1 ? "" : "s"} (not wired in this prototype).`;
}

async function refreshEstimate() {
  const est = await invoke("estimate", {
    providerName: state.provider,
    model: state.model,
    resolution: state.resolution,
    quality: state.quality,
  });
  const value = $("cost-value");
  value.textContent = est.cost === 0 ? "Free" : `$${est.cost.toFixed(3)}`;
  value.classList.toggle("free", est.cost === 0);
  $("cost-detail").textContent = `${est.res_key} · ${est.quality_key}`;
}

function showResult(index) {
  state.current = index;
  const item = state.results[index];
  $("empty").hidden = true;
  $("result").hidden = false;
  $("result-img").src = item.data_url;
  $("result-img").alt = item.prompt;
  $("result-path").textContent = item.path;
  renderStrip();
}

function renderStrip() {
  const strip = $("strip");
  strip.hidden = state.results.length < 2;
  strip.replaceChildren(
    ...state.results.map((item, i) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "thumb";
      btn.title = item.prompt;
      btn.setAttribute("aria-current", String(i === state.current));
      const img = document.createElement("img");
      img.src = item.data_url;
      img.alt = item.prompt;
      btn.append(img);
      btn.addEventListener("click", () => showResult(i));
      return btn;
    }),
  );
}

function setBusy(busy) {
  state.busy = busy;
  $("generate").disabled = busy;
  $("working").hidden = !busy;
}

async function generate(event) {
  event?.preventDefault();
  if (state.busy) return;
  const prompt = $("prompt").value;
  $("error").textContent = "";
  setBusy(true);
  const started = Date.now();
  const tick = setInterval(() => {
    $("elapsed").textContent = `${Math.round((Date.now() - started) / 1000)}s`;
  }, 250);
  try {
    const images = await invoke("generate", {
      providerName: state.provider,
      model: state.model,
      resolution: state.resolution,
      quality: state.quality,
      prompt,
      outputDir: $("output-dir").value,
    });
    for (const img of images) state.results.push({ ...img, prompt });
    showResult(state.results.length - 1);
  } catch (err) {
    $("error").textContent = String(err);
  } finally {
    clearInterval(tick);
    $("elapsed").textContent = "0s";
    setBusy(false);
  }
}

async function init() {
  $("composer").addEventListener("submit", generate);
  $("prompt").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) generate(e);
  });
  try {
    $("output-dir").value = await invoke("default_output_dir");
    state.providers = await invoke("providers");
    await selectProvider(state.providers[0].name);
  } catch (err) {
    $("error").textContent = String(err);
  }
  $("prompt").focus();
}

// Lets the UI be opened in a plain browser for layout work, with canned
// answers in place of the Rust commands.
async function mockInvoke(cmd, args) {
  const model = "stabilityai/stable-diffusion-xl-base-1.0";
  switch (cmd) {
    case "default_output_dir": return "/home/me/Pictures";
    case "providers": return [{
      name: "OVH", default_model: model, status_note: "Anonymous access (2 rpm) — set OVH_AI_ENDPOINTS_ACCESS_TOKEN in .env for more",
      models: [{ name: model, max_input_images: 0, min_input_images: 0 }],
    }];
    case "resolution_choices": return { options: ["1:1"], default: "1:1" };
    case "quality_choices": return { options: ["1K (Free)"], default: "1K (Free)" };
    case "estimate": return { res_key: "1024x1024", quality_key: "1K", cost: 0 };
    case "generate":
      await new Promise((r) => setTimeout(r, 1500));
      throw new Error(`Browser preview: nothing generated for "${args.prompt}"`);
  }
}

init();
