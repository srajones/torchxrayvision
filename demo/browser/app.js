const CANONICAL = [
  "Atelectasis",
  "Consolidation",
  "Infiltration",
  "Pneumothorax",
  "Edema",
  "Emphysema",
  "Fibrosis",
  "Effusion",
  "Pneumonia",
  "Pleural_Thickening",
  "Cardiomegaly",
  "Nodule",
  "Mass",
  "Hernia",
  "Lung Lesion",
  "Fracture",
  "Lung Opacity",
  "Enlarged Cardiomediastinum",
];

const LITTLE_ENDIAN = new Set(["1.2.840.10008.1.2", "1.2.840.10008.1.2.1"]);

const drop = document.querySelector("#drop");
const fileInput = document.querySelector("#file");
const choose = document.querySelector("#choose");
const preview = document.querySelector("#preview");
const stage = document.querySelector("#stage");
const mapCanvas = document.querySelector("#map");
const mapToggle = document.querySelector("#map-toggle");
const mapNote = document.querySelector("#map-note");
const prompt = document.querySelector("#prompt");
const hint = document.querySelector("#hint");
const filename = document.querySelector("#filename");
const meta = document.querySelector("#meta");
const errorBox = document.querySelector("#error");
const status = document.querySelector("#status");
const chips = document.querySelector("#chips");
const empty = document.querySelector("#empty");
const findingsList = document.querySelector("#findings");

const sessions = new Map();
const specs = new Map();
const loadState = new Map();
let models = [];
let film = null;
let findings = null;
let openLabel = null;
let showQuiet = false;
let showMap = true;
let highlight = null;
let classMap = null;
let scoring = false;
let runId = 0;
let ort = null;

function showError(message) {
  if (!message) {
    errorBox.hidden = true;
    errorBox.textContent = "";
    return;
  }
  errorBox.hidden = false;
  errorBox.textContent = message;
}

const READER_NAME = {
  "All cohorts": "All datasets",
  NIH: "NIH",
  PadChest: "PadChest",
  CheXpert: "CheXpert",
  "MIMIC-NB": "MIMIC notes",
  "MIMIC-CH": "MIMIC report",
  RSNA: "RSNA",
  "ResNet 512": "Larger image",
};

function readerName(title) {
  return READER_NAME[title] || title;
}

function labelText(label) {
  return label.replaceAll("_", " ");
}

function cleared(finding) {
  return finding.mean >= 0.5;
}

function splitVote(finding) {
  return !cleared(finding) && finding.over * 2 > finding.votes.length;
}

function formatScore(value) {
  return value.toFixed(2);
}

function sigmoid(value) {
  if (value >= 0) {
    const z = Math.exp(-value);
    return 1 / (1 + z);
  }
  const z = Math.exp(value);
  return z / (1 + z);
}

function opNorm(probability, threshold) {
  if (probability < threshold) return probability / (threshold * 2);
  return 1 - (1 - probability) / ((1 - threshold) * 2);
}

function findingsFromModels(perModel) {
  return CANONICAL.map((label, index) => {
    const votes = [];
    for (const row of perModel) {
      const name = row.model.labels[index];
      const threshold = row.model.op_threshs[index];
      if (!name || threshold == null || Number.isNaN(threshold)) continue;
      const logit = row.logits[index];
      if (logit == null || Number.isNaN(logit)) continue;
      votes.push({
        id: row.model.id,
        title: row.model.title,
        score: opNorm(sigmoid(logit), threshold),
      });
    }
    if (votes.length === 0) return null;
    const mean = votes.reduce((sum, vote) => sum + vote.score, 0) / votes.length;
    const over = votes.filter((vote) => vote.score >= 0.5).length;
    return { label, votes, mean, over };
  })
    .filter((row) => row != null)
    .sort((a, b) => b.mean - a.mean);
}

function normalize(values, maxval) {
  let max = -Infinity;
  for (let i = 0; i < values.length; i++) {
    if (values[i] > max) max = values[i];
  }
  if (max > maxval + 1e-3) {
    throw new Error(`Pixel values peak at ${Math.round(max)}, above the expected bound of ${maxval}.`);
  }
  const out = new Float32Array(values.length);
  for (let i = 0; i < values.length; i++) {
    out[i] = (2 * (values[i] / maxval) - 1) * 1024;
  }
  return out;
}

function centerCrop(data, width, height) {
  const crop = Math.min(width, height);
  const startX = Math.floor(width / 2) - Math.floor(crop / 2);
  const startY = Math.floor(height / 2) - Math.floor(crop / 2);
  const out = new Float32Array(crop * crop);
  for (let y = 0; y < crop; y++) {
    const src = (startY + y) * width + startX;
    out.set(data.subarray(src, src + crop), y * crop);
  }
  return { data: out, size: crop };
}

function clamp(value, low, high) {
  return Math.max(low, Math.min(high, value));
}

function resizeBilinear(src, size, dest) {
  if (size === dest) return src;
  const out = new Float32Array(dest * dest);
  const scale = size / dest;
  for (let y = 0; y < dest; y++) {
    const fy = (y + 0.5) * scale - 0.5;
    const y0 = Math.floor(fy);
    const wy = fy - y0;
    const cy0 = clamp(y0, 0, size - 1);
    const cy1 = clamp(y0 + 1, 0, size - 1);
    for (let x = 0; x < dest; x++) {
      const fx = (x + 0.5) * scale - 0.5;
      const x0 = Math.floor(fx);
      const wx = fx - x0;
      const cx0 = clamp(x0, 0, size - 1);
      const cx1 = clamp(x0 + 1, 0, size - 1);
      const v00 = src[cy0 * size + cx0];
      const v01 = src[cy0 * size + cx1];
      const v10 = src[cy1 * size + cx0];
      const v11 = src[cy1 * size + cx1];
      const top = v00 * (1 - wx) + v01 * wx;
      const bottom = v10 * (1 - wx) + v11 * wx;
      out[y * dest + x] = top * (1 - wy) + bottom * wy;
    }
  }
  return out;
}

function prepareInput(active, resolution) {
  const cropped = centerCrop(active.pixels, active.width, active.height);
  return resizeBilinear(cropped.data, cropped.size, resolution);
}

function looksLikeDicom(buffer, name) {
  const lower = name.toLowerCase();
  if (lower.endsWith(".dcm") || lower.endsWith(".dicom")) return true;
  if (buffer.byteLength < 132) return false;
  const marker = new Uint8Array(buffer, 128, 4);
  return String.fromCharCode(marker[0], marker[1], marker[2], marker[3]) === "DICM";
}

function displayUrl(data, width, height) {
  const sorted = Array.from(data).sort((a, b) => a - b);
  const low = sorted[Math.floor(sorted.length * 0.01)] ?? -1024;
  const high = sorted[Math.floor(sorted.length * 0.99)] ?? 1024;
  const span = high - low || 1;
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d");
  if (!context) throw new Error("Could not draw the image.");
  const image = context.createImageData(width, height);
  for (let i = 0; i < data.length; i++) {
    const gray = Math.max(0, Math.min(255, ((data[i] - low) / span) * 255));
    const offset = i * 4;
    image.data[offset] = gray;
    image.data[offset + 1] = gray;
    image.data[offset + 2] = gray;
    image.data[offset + 3] = 255;
  }
  context.putImageData(image, 0, 0);
  return canvas.toDataURL("image/jpeg", 0.85);
}

function decodeDicom(buffer) {
  const parser = globalThis.dicomParser;
  if (!parser || typeof parser.parseDicom !== "function") {
    throw new Error("DICOM reader did not load. Re-run ./start.sh so vendor files download.");
  }
  const bytes = new Uint8Array(buffer);
  const dataSet = parser.parseDicom(bytes);
  const syntax = dataSet.string("x00020010") || "1.2.840.10008.1.2";
  if (!LITTLE_ENDIAN.has(syntax)) {
    throw new Error("This DICOM is compressed or big-endian. Use PNG, JPG, or uncompressed little-endian DICOM.");
  }
  const photometric = dataSet.string("x00280004") || "";
  if (photometric !== "MONOCHROME1" && photometric !== "MONOCHROME2") {
    throw new Error(`Photometric interpretation ${photometric || "unknown"} is not supported.`);
  }
  const rows = dataSet.uint16("x00280010");
  const cols = dataSet.uint16("x00280011");
  const bitsAllocated = dataSet.uint16("x00280100") || 16;
  const bitsStored = dataSet.uint16("x00280101") || bitsAllocated;
  const pixelRepresentation = dataSet.uint16("x00280103") || 0;
  const samples = dataSet.uint16("x00280002") || 1;
  if (!rows || !cols) throw new Error("DICOM is missing rows or columns.");
  if (samples !== 1) throw new Error("Only single-channel grayscale DICOM is supported.");
  const element = dataSet.elements.x7fe00010;
  if (!element || element.fragments) {
    throw new Error("Encapsulated pixel data needs a codec this page does not include.");
  }
  const count = rows * cols;
  const values = new Float32Array(count);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (bitsAllocated <= 8) {
    if (element.length < count) throw new Error("DICOM pixel data is shorter than the image.");
    for (let i = 0; i < count; i++) {
      let value = bytes[element.dataOffset + i] ?? 0;
      if (pixelRepresentation === 1 && value >= 128) value -= 256;
      values[i] = value;
    }
  } else {
    if (element.length < count * 2) throw new Error("DICOM pixel data is shorter than the image.");
    for (let i = 0; i < count; i++) {
      const offset = element.dataOffset + i * 2;
      values[i] = pixelRepresentation === 1 ? view.getInt16(offset, true) : view.getUint16(offset, true);
    }
  }
  const maxPossible = 2 ** bitsStored - 1;
  if (photometric === "MONOCHROME1") {
    for (let i = 0; i < count; i++) values[i] = maxPossible - values[i];
  }
  const pixels = normalize(values, maxPossible);
  return {
    width: cols,
    height: rows,
    pixels,
    previewUrl: displayUrl(pixels, cols, rows),
    kind: "dicom",
  };
}

async function decodeRaster(file) {
  const bitmap = await createImageBitmap(file);
  const width = bitmap.width;
  const height = bitmap.height;
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context) {
    bitmap.close();
    throw new Error("Could not read that image.");
  }
  context.drawImage(bitmap, 0, 0);
  bitmap.close();
  const image = context.getImageData(0, 0, width, height);
  const raw = new Float32Array(width * height);
  for (let i = 0; i < raw.length; i++) raw[i] = image.data[i * 4] ?? 0;
  return {
    width,
    height,
    pixels: normalize(raw, 255),
    previewUrl: URL.createObjectURL(file),
    kind: "image",
  };
}

async function decodeFilm(file) {
  const buffer = await file.arrayBuffer();
  if (looksLikeDicom(buffer, file.name)) return decodeDicom(buffer);
  return decodeRaster(new File([buffer], file.name, { type: file.type }));
}

function renderStatus() {
  const ready = [...loadState.values()].filter((state) => state === "ready").length;
  const total = models.length;
  status.textContent =
    total === 0 ? "Getting readers ready" : scoring ? "Comparing readers" : `${ready} of ${total} readers ready`;
}

function renderChips() {
  chips.replaceChildren();
  for (const model of models) {
    const state = loadState.get(model.id) || "wait";
    const chip = document.createElement("span");
    chip.className = `chip ${state === "ready" || state === "error" ? state : ""}`;
    const mark = state === "loading" ? "…" : state === "error" ? " ×" : "";
    chip.textContent = readerName(model.title) + mark;
    chips.append(chip);
  }
}

function renderFilm() {
  const hasFilm = film != null;
  prompt.hidden = hasFilm;
  hint.hidden = hasFilm;
  preview.hidden = !hasFilm;
  stage.hidden = !hasFilm;
  if (hasFilm) preview.src = film.previewUrl;
  meta.hidden = !hasFilm;
  if (hasFilm) {
    const kind = film.kind === "dicom" ? "DICOM" : "Image";
    meta.textContent = `${kind} · ${film.width}×${film.height}`;
  }
  choose.textContent = hasFilm ? "Replace" : "Choose image";
  mapToggle.hidden = !classMap;
  mapToggle.textContent = showMap ? "Hide highlight" : "Show highlight";
  const label = highlight ? labelText(highlight) : "";
  mapNote.hidden = !(hasFilm && showMap && classMap && label);
  mapNote.textContent = label
    ? `Where All datasets looked for ${label}. Bright means attention, not a traced lesion.`
    : "";
  paintMap();
}

function appendFinding(list, finding) {
  const item = document.createElement("li");
  item.className = "row";
  const open = openLabel === finding.label;
  const button = document.createElement("button");
  button.type = "button";
  button.setAttribute("aria-expanded", open ? "true" : "false");
  const top = document.createElement("span");
  top.className = "row-top";
  const name = document.createElement("span");
  name.className = highlight === finding.label ? "name on" : "name";
  name.textContent = labelText(finding.label);
  const agree = document.createElement("span");
  agree.className = "agree";
  agree.textContent = `${finding.over} of ${finding.votes.length}`;
  const chev = document.createElement("span");
  chev.className = "chev";
  chev.textContent = open ? "▾" : "▸";
  top.append(name, agree, chev);
  const meter = document.createElement("span");
  meter.className = "meter";
  const track = document.createElement("span");
  track.className = "track";
  const fill = document.createElement("span");
  fill.style.width = `${Math.max(0, Math.min(100, finding.mean * 100))}%`;
  const tick = document.createElement("span");
  tick.className = "tick";
  track.append(fill, tick);
  const score = document.createElement("span");
  score.className = "score";
  score.textContent = formatScore(finding.mean);
  meter.append(track, score);
  button.append(top, meter);
  button.addEventListener("click", () => {
    openLabel = open ? null : finding.label;
    highlight = finding.label;
    renderFindings();
    paintMap();
  });
  item.append(button);
  if (open) {
    const votes = document.createElement("ul");
    votes.className = "votes";
    for (const vote of finding.votes) {
      const row = document.createElement("li");
      const voteName = document.createElement("span");
      voteName.className = "vote-name";
      voteName.textContent = readerName(vote.title);
      const voteTrack = document.createElement("span");
      voteTrack.className = "track";
      const voteFill = document.createElement("span");
      voteFill.className = "ok";
      voteFill.style.width = `${Math.max(0, Math.min(100, vote.score * 100))}%`;
      const voteTick = document.createElement("span");
      voteTick.className = "tick";
      voteTrack.append(voteFill, voteTick);
      const voteScore = document.createElement("span");
      voteScore.className = "flag";
      voteScore.textContent = vote.score >= 0.5 ? "flagged" : "under";
      row.append(voteName, voteTrack, voteScore);
      votes.append(row);
    }
    item.append(votes);
  }
  list.append(item);
}

function renderFindings() {
  const ready = [...loadState.values()].filter((state) => state === "ready").length;
  if (findings == null) {
    findingsList.hidden = true;
    findingsList.replaceChildren();
    empty.hidden = false;
    empty.textContent =
      ready === 0
        ? "The first reader is loading. You can drop a film now."
        : film
          ? "Comparing the readers that are ready. The others will join as they finish."
          : "Drop a film. Each reader votes only on findings it was trained to see.";
    return;
  }
  empty.hidden = true;
  findingsList.hidden = false;
  findingsList.replaceChildren();

  const standouts = findings.filter(cleared);
  const splits = findings.filter(splitVote);
  const lead = findings[0];
  const showLeadAlone = standouts.length === 0 && splits.length === 0;
  const featured = showLeadAlone ? [lead] : standouts;
  const quiet = findings.filter(
    (finding) => !cleared(finding) && !splitVote(finding) && !featured.includes(finding),
  );

  const summary = document.createElement("li");
  summary.className = "summary";
  const eyebrow = document.createElement("p");
  eyebrow.className = "eyebrow";
  eyebrow.textContent = cleared(lead) ? "Cleared the cutoff" : "Closest";
  const headline = document.createElement("p");
  headline.className = "headline";
  headline.textContent = labelText(lead.label);
  const detail = document.createElement("p");
  detail.className = "detail";
  detail.textContent = `${lead.over} of ${lead.votes.length} readers flagged it. Score ${formatScore(lead.mean)}.${
    cleared(lead) ? "" : " That is under the usual cutoff."
  }`;
  summary.append(eyebrow, headline, detail);
  findingsList.append(summary);

  if (!showLeadAlone && standouts.length > 0) {
    const heading = document.createElement("li");
    heading.className = "group";
    heading.textContent = "At or above the cutoff";
    findingsList.append(heading);
  }
  for (const finding of featured) appendFinding(findingsList, finding);

  if (splits.length > 0) {
    const heading = document.createElement("li");
    heading.className = "group";
    heading.textContent = "Split vote";
    findingsList.append(heading);
    for (const finding of splits) appendFinding(findingsList, finding);
  }

  if (quiet.length > 0) {
    const toggle = document.createElement("li");
    const button = document.createElement("button");
    button.type = "button";
    button.className = "quiet-toggle";
    button.textContent = showQuiet ? "Hide quieter findings" : `Show ${quiet.length} quieter findings`;
    button.addEventListener("click", () => {
      showQuiet = !showQuiet;
      renderFindings();
    });
    toggle.append(button);
    findingsList.append(toggle);
    if (showQuiet) {
      for (const finding of quiet) appendFinding(findingsList, finding);
    }
  }
}

function render() {
  renderStatus();
  renderChips();
  renderFilm();
  renderFindings();
}

async function score() {
  if (!film || !ort || sessions.size === 0) return;
  const active = film;
  const token = ++runId;
  scoring = true;
  renderStatus();
  try {
    const bySize = new Map();
    const perModel = [];
    for (const [id, session] of sessions) {
      const model = specs.get(id);
      if (!model) continue;
      let input = bySize.get(model.resolution);
      if (!input) {
        input = prepareInput(active, model.resolution);
        bySize.set(model.resolution, input);
      }
      const tensor = new ort.Tensor("float32", input, [1, 1, model.resolution, model.resolution]);
      const output = await session.run({ input: tensor });
      const logits = output.logits?.data ?? Object.values(output)[0]?.data;
      if (!logits) throw new Error(`${model.title} returned no scores.`);
      perModel.push({ model, logits });
      if (model.map && output.cam?.dims) {
        classMap = { side: output.cam.dims[2], values: output.cam.data };
      }
    }
    if (token !== runId || active !== film) return;
    findings = findingsFromModels(perModel);
    if (!highlight || !findings.some((finding) => finding.label === highlight)) {
      highlight = findings[0]?.label ?? null;
    }
  } catch (error) {
    if (token === runId) showError(error instanceof Error ? error.message : "Scoring failed.");
  } finally {
    if (token === runId) {
      scoring = false;
      render();
    }
  }
}

async function takeFile(file) {
  if (!file) return;
  showError("");
  try {
    const next = await decodeFilm(file);
    if (film && film.previewUrl.startsWith("blob:")) URL.revokeObjectURL(film.previewUrl);
    film = next;
    filename.textContent = file.name;
    findings = null;
    openLabel = null;
    showQuiet = false;
    highlight = null;
    classMap = null;
    render();
    void score();
  } catch (error) {
    showError(error instanceof Error ? error.message : "Could not read that file.");
  }
}

function openPicker() {
  fileInput.click();
}

drop.addEventListener("click", openPicker);
drop.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    openPicker();
  }
});
drop.addEventListener("dragover", (event) => {
  event.preventDefault();
  drop.classList.add("over");
});
drop.addEventListener("dragleave", () => drop.classList.remove("over"));
drop.addEventListener("drop", (event) => {
  event.preventDefault();
  drop.classList.remove("over");
  void takeFile(event.dataTransfer.files[0]);
});
choose.addEventListener("click", openPicker);
fileInput.addEventListener("change", () => {
  void takeFile(fileInput.files?.[0]);
  fileInput.value = "";
});

async function loadOne(model) {
  loadState.set(model.id, "loading");
  render();
  const url = new URL(`./models/${model.file}`, window.location.href).href;
  const session = await ort.InferenceSession.create(url, { executionProviders: ["wasm"] });
  sessions.set(model.id, session);
  specs.set(model.id, model);
  loadState.set(model.id, "ready");
  render();
  if (film) void score();
}

async function boot() {
  try {
    ort = await import("./vendor/ort.wasm.min.mjs");
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.simd = true;
    const response = await fetch("./models/registry.json");
    if (!response.ok) throw new Error("Model registry did not load.");
    const registry = await response.json();
    models = registry.models;
    for (const model of models) loadState.set(model.id, "wait");
    render();
    const primary = models.find((model) => model.role === "primary") ?? models[0];
    const rest = models.filter((model) => model !== primary);
    if (primary) await loadOne(primary);
    for (const model of rest) await loadOne(model);
  } catch (error) {
    showError(error instanceof Error ? error.message : "Could not start the networks.");
    render();
  }
}

function paintMap() {
  if (!mapCanvas || !preview) return;
  const boxW = preview.clientWidth;
  const boxH = preview.clientHeight;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.max(1, Math.round(boxW * dpr));
  const height = Math.max(1, Math.round(boxH * dpr));
  if (mapCanvas.width !== width || mapCanvas.height !== height) {
    mapCanvas.width = width;
    mapCanvas.height = height;
  }
  const ctx = mapCanvas.getContext("2d");
  if (!ctx) return;
  ctx.clearRect(0, 0, width, height);
  if (!film || !showMap || !classMap || !highlight || boxW < 2 || boxH < 2) return;
  const index = CANONICAL.indexOf(highlight);
  if (index < 0) return;
  const side = classMap.side;
  const cells = side * side;
  const channel = classMap.values.subarray(index * cells, (index + 1) * cells);
  let max = 0;
  for (let i = 0; i < channel.length; i++) if (channel[i] > max) max = channel[i];
  if (max < 1e-6) return;
  const scaleFit = Math.min(width / film.width, height / film.height);
  const viewW = film.width * scaleFit;
  const viewH = film.height * scaleFit;
  const viewX = (width - viewW) / 2;
  const viewY = (height - viewH) / 2;
  const crop = Math.min(film.width, film.height);
  const startX = Math.floor(film.width / 2) - Math.floor(crop / 2);
  const startY = Math.floor(film.height / 2) - Math.floor(crop / 2);
  const grid = 64;
  const off = document.createElement("canvas");
  off.width = grid;
  off.height = grid;
  const offCtx = off.getContext("2d");
  const image = offCtx.createImageData(grid, grid);
  const pixels = image.data;
  for (let y = 0; y < grid; y++) {
    const gy = ((y + 0.5) / grid) * side - 0.5;
    const y0 = Math.max(0, Math.min(side - 1, Math.floor(gy)));
    const y1 = Math.max(0, Math.min(side - 1, y0 + 1));
    const wy = gy - Math.floor(gy);
    for (let x = 0; x < grid; x++) {
      const gx = ((x + 0.5) / grid) * side - 0.5;
      const x0 = Math.max(0, Math.min(side - 1, Math.floor(gx)));
      const x1 = Math.max(0, Math.min(side - 1, x0 + 1));
      const wx = gx - Math.floor(gx);
      const value =
        ((channel[y0 * side + x0] * (1 - wx) + channel[y0 * side + x1] * wx) * (1 - wy) +
          (channel[y1 * side + x0] * (1 - wx) + channel[y1 * side + x1] * wx) * wy) /
        max;
      const alpha = value <= 0.35 ? 0 : ((value - 0.35) / 0.65) * 0.72;
      const offset = (y * grid + x) * 4;
      pixels[offset] = 224;
      pixels[offset + 1] = 164;
      pixels[offset + 2] = 90;
      pixels[offset + 3] = Math.round(alpha * 255);
    }
  }
  offCtx.putImageData(image, 0, 0);
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(
    off,
    viewX + startX * scaleFit,
    viewY + startY * scaleFit,
    crop * scaleFit,
    crop * scaleFit,
  );
}

mapToggle.addEventListener("click", () => {
  showMap = !showMap;
  render();
});
preview.addEventListener("load", paintMap);
new ResizeObserver(paintMap).observe(preview);

render();
void boot();
