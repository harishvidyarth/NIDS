/* NIDS Network Traffic Analyzer — frontend only.
 * Talks only to the existing backend API (/api/*) with the exact same
 * endpoints/request bodies the previous UI used — nothing here changes
 * backend behavior. Every value shown either comes from a real API
 * response or is rendered as N/A — nothing is fabricated. */

const API = "";
const el = (id) => document.getElementById(id);

/* ---- Static feature-schema reference data ----
 * Names copied verbatim from backend/prediction/features.py's
 * TRAINING_FEATURES (the 77 features the existing ANN was trained on)
 * and its ID_COLUMNS_ALIASES concept — grouped only for display. Groups
 * are a presentation choice; the feature list itself is exactly the
 * project's real schema, not invented. */
const FEATURE_GROUPS = {
  "FLOW IDENTIFICATION": [
    "Flow ID", "Source IP", "Source Port", "Destination IP",
    "Destination Port", "Protocol", "Timestamp",
  ],
  "FLOW STATISTICS": [
    "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets",
    "Flow Bytes/s", "Flow Packets/s", "Fwd Packets/s", "Bwd Packets/s",
    "Down/Up Ratio", "Subflow Fwd Packets", "Subflow Fwd Bytes",
    "Subflow Bwd Packets", "Subflow Bwd Bytes", "act_data_pkt_fwd",
    "min_seg_size_forward",
  ],
  "PACKET LENGTH": [
    "Fwd Packet Length Max", "Fwd Packet Length Min", "Fwd Packet Length Mean",
    "Fwd Packet Length Std", "Bwd Packet Length Max", "Bwd Packet Length Min",
    "Bwd Packet Length Mean", "Bwd Packet Length Std", "Min Packet Length",
    "Max Packet Length", "Packet Length Mean", "Packet Length Std",
    "Packet Length Variance", "Average Packet Size", "Avg Fwd Segment Size",
    "Avg Bwd Segment Size", "Fwd Header Length", "Bwd Header Length",
  ],
  "TCP FLAGS": [
    "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags",
    "FIN Flag Count", "SYN Flag Count", "RST Flag Count", "PSH Flag Count",
    "ACK Flag Count", "URG Flag Count", "CWE Flag Count", "ECE Flag Count",
  ],
  "TIMING (IAT / ACTIVE / IDLE)": [
    "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min",
    "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
    "Active Mean", "Active Std", "Active Max", "Active Min",
    "Idle Mean", "Idle Std", "Idle Max", "Idle Min",
  ],
  "BULK / WINDOW": [
    "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate",
    "Init_Win_bytes_forward", "Init_Win_bytes_backward",
  ],
};
// Matches backend/temporal/validate.py's checks dict keys exactly, in the
// display order requested (task section 47).
const CHECK_ORDER = [
  ["current_state", "Current State"],
  ["timestamps", "Timestamps"],
  ["windows", "10s Windows"],
  ["features", "28 Features"],
  ["transitions", "Transitions"],
  ["sequences", "Sequences"],
  ["chronological_split", "Chronological Split"],
  ["leakage", "Data Leakage"],
  ["missing_data", "Missing Data"],
  ["duplicates", "Duplicates"],
];
function statusIcon(status) {
  if (status === "PASS") return "&#10003;";
  if (status === "WARNING") return "&#9888;";
  if (status === "FAIL") return "&#10007;";
  return "&#8212;";
}

const MODEL_FEATURE_COUNT = 77; // FLOW IDENTIFICATION's Destination Port is also a model feature; rest of that group is not.
// Names in FLOW IDENTIFICATION that are pure metadata, never fed to the ANN
// (Destination Port is the one exception — it's a real model feature too).
const ID_ONLY_FIELDS = new Set(["Flow ID", "Source IP", "Source Port", "Destination IP", "Protocol", "Timestamp"]);

/* ---- Global state (mirrors real backend state only) ---- */
let mode = "live";
let backendConnected = null;
let liveSnap = null;
let uploadStatus = null;
let uploadSessionId = null;
let selectedUploadFile = null;
let temporalStatus = { stage: "IDLE", error: null, result: null };
let activeTableTab = "packets";
let activeDetailTab = "capture";
let pollTimer = null, packetTimer = null, uploadPollTimer = null;
let temporalPolling = false;
let validationStatus = { stage: "NOT_VALIDATED", error: null, report: null };
let validationPolling = false;
let lstmStatus = { stage: "idle", rows_processed: 0, cache_state: "unknown", epoch: 0 };
let lstmReport = null;
let lstmForecast = null;
let multistepForecast = null;
let lstmPolling = false;
let activeInnerTab = "overview";
let activeTemporalSubtab = "states";
let selectedValidationCheck = null;
let explanationResults = {};
let explanationPolling = new Set();
let temporalStatesData = null;   // { session, source, window_size_seconds, rows: [...] }
let temporalStatesKey = null;    // session name + row count, to avoid redundant refetch
let packetRateHistory = [];       // [{ t: ms, packets }] for the capture sparkline
let responseCapabilities = null;
let responseToken = null;
let responseScan = null;
let responsePlan = null;
let responseAction = null;
let responseHistory = [];
let responseAudit = [];
let xdrEnrichment = null;
let xdrGraph = null;
let xdrTriage = null;
let xdrHits = [];
let xdrAudit = [];
let xdrPlan = null;
let xdrAction = null;

async function api(path, opts) {
  const res = await fetch(API + path, opts);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    const detail = body.detail;
    const message = detail && typeof detail === "object" ? (detail.message || JSON.stringify(detail)) : detail;
    const error = new Error(message || `${res.status} ${res.statusText}`);
    error.status = res.status;
    error.detail = detail;
    throw error;
  }
  return res.json();
}

function fmtDuration(seconds) {
  if (seconds === null || seconds === undefined) return "N/A";
  const s = Math.floor(seconds);
  return `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;
}
function fmtTime(epochSeconds) {
  if (!epochSeconds) return "N/A";
  return new Date(epochSeconds * 1000).toLocaleTimeString();
}
function fmtBytes(n) {
  if (n === null || n === undefined) return "N/A";
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(2)} MB`;
}
function escapeHtml(v) {
  if (v === null || v === undefined) return "";
  return String(v).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}
function pickField(flow, candidates, fallback) {
  for (const c of candidates) {
    if (flow[c] !== undefined && flow[c] !== null && flow[c] !== "") return flow[c];
  }
  return fallback;
}
function na(v) { return (v === null || v === undefined || v === "") ? "N/A" : v; }

function setBackendConnected(ok) {
  if (ok === backendConnected) return;
  backendConnected = ok;
  const label = ok ? "CONNECTED" : "UNREACHABLE";
  el("hdr-backend-val").textContent = label;
  el("hdr-backend").querySelector(".status-dot").className = "status-dot " + (ok ? "status-stopped" : "status-error");
  el("sb-backend").innerHTML = (ok ? "&#9679; BACKEND CONNECTED" : "&#9679; BACKEND UNREACHABLE");
  el("sb-backend").style.color = ok ? "" : "var(--red)";
}

function showError(message, severity) {
  const banner = el("error-banner");
  if (!message) { banner.style.display = "none"; return; }
  const warn = severity === "warning";
  banner.classList.toggle("is-warning", warn);
  el("error-banner-text").textContent = message;
  banner.querySelector(".error-banner-icon").innerHTML = warn ? "&#9888;" : "&#10007;";
  banner.querySelector(".error-banner-title").textContent = warn ? "WARNING" : "ERROR";
  banner.style.display = "";
}

/* ==================== Mode switching ==================== */
function switchMode(newMode) {
  if (mode !== newMode) {
    responsePlan = null;
    responseAction = null;
    responseAudit = [];
  }
  mode = newMode;
  const isLive = mode === "live";
  el("mode-live").className = "mode-btn" + (isLive ? " mode-btn-active" : "");
  el("mode-upload").className = "mode-btn" + (!isLive ? " mode-btn-active" : "");
  el("mode-live").innerHTML = (isLive ? "&#9679;" : "&#9675;") + "&nbsp;LIVE CAPTURE";
  el("mode-upload").innerHTML = (!isLive ? "&#9679;" : "&#9675;") + "&nbsp;FILE ANALYSIS";
  el("live-controls").style.display = isLive ? "" : "none";
  el("upload-controls").style.display = isLive ? "none" : "";
  showError(null);
  renderAll();
}

/* ==================== Live capture (unchanged API calls) ==================== */
async function loadInterfaces() {
  const sel = el("iface-select");
  sel.disabled = true;
  sel.innerHTML = '<option value="">Loading interfaces…</option>';
  try {
    const data = await api("/api/interfaces");
    sel.innerHTML = "";
    for (const iface of data.interfaces) {
      const opt = document.createElement("option");
      opt.value = iface.device;
      opt.textContent = iface.name || iface.device;
      opt.title = iface.description || iface.device;
      sel.appendChild(opt);
    }
    if (data.interfaces.length === 0) {
      sel.innerHTML = '<option value="">No capture interfaces found</option>';
    } else {
      sel.disabled = false;
    }
  } catch (e) {
    sel.innerHTML = `<option value="">Interface API error: ${escapeHtml(e.message)}</option>`;
  }
}

async function startCapture() {
  const captureAll = el("iface-all") && el("iface-all").checked;
  const iface = captureAll ? "all" : el("iface-select").value;
  if (!iface) return;
  el("btn-start").disabled = true;
  try {
    await api("/api/capture/start", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ interface: iface }),
    });
    packetsPage = 0;
    packetsTotal = null;
    startPacketPolling();
  } catch (e) {
    alert("Failed to start capture: " + e.message);
    el("btn-start").disabled = false;
  }
  refreshPipeline();
}

async function stopCapture() {
  el("btn-stop").disabled = true;
  try {
    await api("/api/capture/stop", { method: "POST" });
  } catch (e) {
    alert("Failed to stop capture: " + e.message);
  }
  refreshPipeline();
}

async function runExtract() {
  el("btn-extract").disabled = true;
  try {
    await api("/api/extract", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({}),
    });
  } catch (e) {
    alert("Failed to start extraction: " + e.message);
    el("btn-extract").disabled = false;
  }
  refreshPipeline();
}

async function runPredict() {
  el("btn-predict").disabled = true;
  try {
    await api("/api/predict", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({}),
    });
  } catch (e) {
    alert("Failed to start prediction: " + e.message);
    el("btn-predict").disabled = false;
  }
  refreshPipeline();
}

async function resetPipeline() {
  try {
    await api("/api/pipeline/reset", { method: "POST" });
  } catch (e) {
    alert("Reset failed: " + e.message);
  }
  stopPacketPolling();
  liveSnap = null;
  lastPackets = [];
  packetsPage = 0;
  packetsTotal = null;
  wasCapturing = false;
  temporalStatus = { stage: "IDLE", error: null, result: null };
  validationStatus = { stage: "NOT_VALIDATED", error: null, report: null };
  selectedValidationCheck = null;
  responsePlan = null;
  responseAction = null;
  responseAudit = [];
  refreshPipeline();
}

// Packet table is paginated server-side (task: handle 50,000+ captured
// packets without ever rendering them all into the DOM at once — see
// capture.read_packets()'s tshark frame-number range filter). Only the
// current page's rows are ever fetched or rendered.
let packetsPage = 0;
const PACKETS_PAGE_SIZE = 100;
let packetsTotal = null;
let lastPackets = [];
let wasCapturing = false;

async function pollPackets() {
  try {
    const offset = packetsPage * PACKETS_PAGE_SIZE;
    const data = await api(`/api/capture/packets?offset=${offset}&limit=${PACKETS_PAGE_SIZE}`);
    lastPackets = data.packets;
    packetsTotal = data.total_packets;
    if (mode === "live" && activeTableTab === "packets") renderPacketsTable();
  } catch (e) { /* transient */ }
}
function startPacketPolling() {
  stopPacketPolling();
  packetTimer = setInterval(pollPackets, 1500);
  pollPackets();
}
function stopPacketPolling() {
  if (packetTimer) clearInterval(packetTimer);
  packetTimer = null;
}
function goToPacketsPage(delta) {
  const totalPages = packetsTotal ? Math.max(1, Math.ceil(packetsTotal / PACKETS_PAGE_SIZE)) : packetsPage + 2;
  packetsPage = Math.max(0, Math.min(packetsPage + delta, totalPages - 1));
  pollPackets();
}

async function refreshPipeline() {
  try {
    liveSnap = await api("/api/pipeline");
    setBackendConnected(true);
  } catch (e) {
    setBackendConnected(false);
    liveSnap = null;
  }
  await syncTemporalAndValidationStatus();
  renderAll();
}

// Keeps temporalStatus/validationStatus in sync with the server on the
// same cadence as the rest of the UI, so a page reload (or first load)
// picks up an already-completed temporal dataset/validation from this
// server process instead of only updating when this page session itself
// triggered the action. Skipped while an active prepare/validate action
// owns fast polling of the same endpoints, to avoid duplicate requests.
async function syncTemporalAndValidationStatus() {
  if (!temporalPolling) {
    try { temporalStatus = await api("/api/temporal/status"); } catch (e) { /* leave last known */ }
  }
  try { await loadTemporalStates(); } catch (e) { /* leave last known */ }
  if (!validationPolling) {
    try { validationStatus = await api("/api/temporal/validate/status"); } catch (e) { /* leave last known */ }
  }
  if (!lstmPolling) {
    try {
      lstmStatus = await api("/api/lstm/status");
      if (lstmStatus.stage === "completed" && !lstmReport) lstmReport = await api("/api/lstm/report");
    } catch (e) { /* leave last known */ }
  }
}

function startPolling() {
  if (pollTimer) clearInterval(pollTimer);
  pollTimer = setInterval(refreshPipeline, 1200);
  refreshPipeline();
}

/* ==================== File upload (unchanged API calls) ==================== */
async function analyzeFile() {
  if (!selectedUploadFile) return;
  showError(null);
  el("btn-analyze").disabled = true;
  const formData = new FormData();
  formData.append("file", selectedUploadFile);
  let resp;
  try {
    const res = await fetch(API + "/api/upload", { method: "POST", body: formData });
    resp = await res.json();
    if (!res.ok) throw new Error(resp.detail || `${res.status}`);
  } catch (e) {
    showError("Upload rejected: " + e.message);
    el("btn-analyze").disabled = false;
    return;
  }
  uploadSessionId = resp.session_id;
  uploadStatus = null;
  temporalStatus = { stage: "IDLE", error: null, result: null };
  startUploadPolling();
}

async function pollUploadStatus() {
  if (!uploadSessionId) return;
  try {
    uploadStatus = await api(`/api/upload/${uploadSessionId}/status`);
    setBackendConnected(true);
  } catch (e) {
    setBackendConnected(false);
  }
  if (mode === "upload") renderAll();
  if (uploadStatus && (uploadStatus.stage === "PREDICTION_COMPLETED" || uploadStatus.stage === "ERROR")) {
    stopUploadPolling();
    el("btn-analyze").disabled = false;
  }
}
function startUploadPolling() {
  stopUploadPolling();
  uploadPollTimer = setInterval(pollUploadStatus, 1000);
  pollUploadStatus();
}
function stopUploadPolling() {
  if (uploadPollTimer) clearInterval(uploadPollTimer);
  uploadPollTimer = null;
}
function downloadProcessedCsv() {
  if (mode === "upload" && uploadSessionId) {
    window.location.href = `${API}/api/upload/${uploadSessionId}/download`;
  }
}

/* ==================== Temporal dataset (unchanged API calls) ==================== */
async function prepareTemporalDataset() {
  const csvPath = currentExtractionOutputCsv();
  if (!csvPath) return;
  el("btn-temporal").disabled = true;
  temporalPolling = true;
  try {
    await api("/api/temporal/prepare", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ csv_path: csvPath }),
    });
  } catch (e) {
    temporalStatus = { stage: "ERROR", error: e.message, result: null };
    temporalPolling = false;
    renderAll();
    return;
  }
  pollTemporalStatus();
}
async function pollTemporalStatus() {
  try {
    temporalStatus = await api("/api/temporal/status");
  } catch (e) {
    temporalStatus = { stage: "ERROR", error: "Status unreachable: " + e.message, result: null };
  }
  if (temporalStatus.stage === "PREPARING") {
    setTimeout(pollTemporalStatus, 1000);
    renderAll();
    return;
  }
  temporalPolling = false;
  renderAll();
}

/* ==================== Temporal dataset validation (new: /api/temporal/validate) ==================== */
async function runValidation() {
  el("btn-validate").disabled = true;
  validationPolling = true;
  selectedValidationCheck = null;
  try {
    await api("/api/temporal/validate", { method: "POST" });
  } catch (e) {
    validationStatus = { stage: "ERROR", error: e.message, report: null };
    validationPolling = false;
    renderAll();
    return;
  }
  pollValidationStatus();
}
async function pollValidationStatus() {
  try {
    validationStatus = await api("/api/temporal/validate/status");
  } catch (e) {
    validationStatus = { stage: "ERROR", error: "Status unreachable: " + e.message, report: null };
  }
  if (validationStatus.stage === "VALIDATING") {
    setTimeout(pollValidationStatus, 1000);
    renderAll();
    return;
  }
  validationPolling = false;
  renderAll();
}
function downloadValidationReport() {
  window.location.href = `${API}/api/temporal/validate/report`;
}

async function startLstmTraining() {
  lstmPolling = true;
  lstmReport = null;
  lstmForecast = null;
  try {
    lstmStatus = await api("/api/lstm/train", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ force_rebuild: false }),
    });
    pollLstmStatus();
  } catch (e) {
    lstmStatus = { stage: "error", error: e.message };
    lstmPolling = false;
    renderAll();
  }
}
async function pollLstmStatus() {
  try { lstmStatus = await api("/api/lstm/status"); }
  catch (e) { lstmStatus = { stage: "error", error: e.message }; }
  if (!["completed", "error"].includes(lstmStatus.stage)) {
    setTimeout(pollLstmStatus, 1000);
  } else {
    lstmPolling = false;
    if (lstmStatus.stage === "completed") {
      try { lstmReport = await api("/api/lstm/report"); } catch (e) { lstmStatus.error = e.message; }
    }
  }
  renderAll();
}
async function runLstmForecast() {
  try { lstmForecast = await api("/api/lstm/forecast", { method: "POST" }); }
  catch (e) { lstmStatus.error = e.message; }
  renderAll();
}
async function runMultistepForecast() {
  el("btn-multistep-forecast").disabled = true;
  try {
    multistepForecast = await api("/api/lstm/forecast/multistep", { method: "POST" });
    el("multi-error").style.display = "none";
  } catch (e) {
    el("multi-error").style.display = "";
    el("multi-error").textContent = e.message;
  }
  el("btn-multistep-forecast").disabled = false;
  renderMultistepTab();
}
function downloadLstmReport() { window.location.href = `${API}/api/lstm/report`; }

function currentExtractionOutputCsv() {
  if (mode === "live") return liveSnap && liveSnap.extraction ? liveSnap.extraction.output_csv : null;
  if (mode === "upload") return uploadStatus && uploadStatus.stage === "PREDICTION_COMPLETED" ? uploadStatus.processed_csv_path : null;
  return null;
}

/* ==================== Pipeline strip (stages 1-3 live-only; stage 4 always) ==================== */
function stageClass(stage, forStage) {
  const map = {
    capture: { running: "CAPTURING", done: ["CAPTURE_COMPLETED", "EXTRACTING", "EXTRACTION_COMPLETED", "PREDICTING", "PREDICTION_COMPLETED"] },
    extract: { running: "EXTRACTING", done: ["EXTRACTION_COMPLETED", "PREDICTING", "PREDICTION_COMPLETED"] },
    predict: { running: "PREDICTING", done: ["PREDICTION_COMPLETED"] },
  };
  const m = map[forStage];
  if (stage === "ERROR") return "state-failed";
  if (stage === m.running) return "state-running";
  if (m.done.includes(stage)) return "state-completed";
  return "";
}
function uploadStageClass(stage, forStage, inputType) {
  // CSV uploads never go through EXTRACTING (predict_csv() runs directly on
  // the uploaded CSV) — showing that stage as "completed" would falsely
  // imply CICFlowMeter ran, so it's marked "skipped" instead, distinctly.
  if (forStage === "extract" && inputType === "csv") return "state-skipped";

  const doneAfter = {
    capture: ["VALIDATING", "EXTRACTING", "PREDICTING", "PREDICTION_COMPLETED"],
    extract: ["PREDICTING", "PREDICTION_COMPLETED"],
    predict: ["PREDICTION_COMPLETED"],
  };
  const runningMap = { capture: "FILE_UPLOADED", extract: "EXTRACTING", predict: "PREDICTING" };
  if (stage === "ERROR") return "state-failed";
  if (forStage === "capture" && stage === "VALIDATING") return "state-running";
  if (stage === runningMap[forStage]) return "state-running";
  if (doneAfter[forStage].includes(stage)) return "state-completed";
  return "";
}
function stageIcon(cls) {
  if (cls === "state-completed") return "&#10003;";
  if (cls === "state-running") return "&#9679;";
  if (cls === "state-failed") return "&#10007;";
  if (cls === "state-skipped") return "&#8212;";
  return "&#9675;";
}

function renderPipelineStrip() {
  const stage = mode === "live" ? (liveSnap ? liveSnap.stage : "IDLE") : (uploadStatus ? uploadStatus.stage : "IDLE");
  const inputType = mode === "upload" && uploadStatus ? uploadStatus.input_type : null;
  const classifier = mode === "live" ? (s, f) => stageClass(s, f) : (s, f) => uploadStageClass(s, f, inputType);
  for (const s of ["capture", "extract", "predict"]) {
    const cls = classifier(stage, s);
    document.querySelector(`.pipeline-stage[data-stage="${s}"]`).className = "pipeline-stage " + cls;
    el(`icon-${s}`).innerHTML = stageIcon(cls);
  }
  el("label-capture").textContent = mode === "live" ? "PCAP CAPTURE" : "FILE INPUT";

  let tCls = "";
  if (temporalStatus.stage === "ERROR") tCls = "state-failed";
  else if (temporalStatus.stage === "PREPARING") tCls = "state-running";
  else if (temporalStatus.stage === "COMPLETED") tCls = "state-completed";
  document.querySelector('.pipeline-stage[data-stage="temporal"]').className = "pipeline-stage " + tCls;
  el("icon-temporal").innerHTML = stageIcon(tCls);

  const errMsg = mode === "live"
    ? (liveSnap && liveSnap.error ? liveSnap.error : (temporalStatus.stage === "ERROR" ? temporalStatus.error : ""))
    : (uploadStatus && uploadStatus.error ? uploadStatus.error : (temporalStatus.stage === "ERROR" ? temporalStatus.error : ""));
  const capWarn = mode === "live" && liveSnap && liveSnap.capture ? liveSnap.capture.warning : "";
  if (errMsg) showError(errMsg, "error");
  else if (capWarn) showError(capWarn, "warning");
  else showError(null);
}

/* ==================== Table tabs ==================== */
function switchTableTab(tab) {
  activeTableTab = tab;
  document.querySelectorAll(".table-tab").forEach((b) => b.classList.toggle("table-tab-active", b.dataset.tableTab === tab));
  for (const t of ["packets", "flows", "predictions", "featurevalues", "temporal"]) {
    el(`tablewrap-${t}`).style.display = t === tab ? "" : "none";
  }
  el("predictions-state-filter-wrap").style.display = tab === "predictions" ? "" : "none";
  renderActiveTable();
}

function currentFlows() {
  const pred = mode === "live" ? (liveSnap && liveSnap.prediction) : (uploadStatus && uploadStatus.prediction);
  return pred && pred.flows ? pred.flows : null;
}

function applyTextFilters(rows, getFilterText, getSearchText) {
  const filterVal = el("table-filter").value.trim().toLowerCase();
  const searchVal = el("table-search").value.trim().toLowerCase();
  return rows.filter((r) => {
    if (filterVal && !getFilterText(r).toLowerCase().includes(filterVal)) return false;
    if (searchVal && !getSearchText(r).toLowerCase().includes(searchVal)) return false;
    return true;
  });
}

function renderPacketsTable() {
  if (mode === "upload") {
    el("packets-tbody").innerHTML = '<tr class="empty-row"><td colspan="9">N/A — packet-level data is not exposed by the current API for uploaded files (only capture mode reads live packets via tshark).</td></tr>';
    el("table-row-count").textContent = "0 rows";
    renderPacketsPagination();
    return;
  }
  const rows = applyTextFilters(lastPackets || [],
    (p) => `${p.protocol} ${p.info}`, (p) => `${p.source} ${p.destination} ${p.src_port} ${p.dst_port}`);
  if (rows.length === 0) {
    el("packets-tbody").innerHTML = '<tr class="empty-row"><td colspan="9">No packets captured yet.</td></tr>';
    el("table-row-count").textContent = "0 rows";
    renderPacketsPagination();
    return;
  }
  el("table-row-count").textContent = `${rows.length} rows (this page)`;
  el("packets-tbody").innerHTML = rows.map((p) => `
    <tr>
      <td>${p.no}</td><td>${p.time === null ? "" : p.time.toFixed(6)}</td>
      <td>${escapeHtml(p.source)}</td><td>${escapeHtml(p.destination)}</td>
      <td class="proto-${escapeHtml(p.protocol)}">${escapeHtml(p.protocol)}</td>
      <td>${escapeHtml(p.src_port)}</td><td>${escapeHtml(p.dst_port)}</td>
      <td>${p.length ?? ""}</td><td class="col-info">${escapeHtml(p.info)}</td>
    </tr>`).join("");
  renderPacketsPagination();
}

function renderPacketsPagination() {
  const offset = packetsPage * PACKETS_PAGE_SIZE;
  const shown = (lastPackets || []).length;
  const rangeStart = shown > 0 ? offset + 1 : 0;
  const rangeEnd = offset + shown;
  const totalLabel = packetsTotal != null ? packetsTotal.toLocaleString() : "?";
  el("packets-pagination-label").textContent = `Rows ${rangeStart}-${rangeEnd} of ${totalLabel}`;
  const totalPages = packetsTotal ? Math.max(1, Math.ceil(packetsTotal / PACKETS_PAGE_SIZE)) : (packetsPage + 1);
  el("packets-page-label").textContent = `Page ${packetsPage + 1} of ${packetsTotal ? totalPages : "?"}`;
  el("btn-packets-prev").disabled = mode !== "live" || packetsPage <= 0;
  el("btn-packets-next").disabled = mode !== "live" || (packetsTotal != null && offset + PACKETS_PAGE_SIZE >= packetsTotal) || shown < PACKETS_PAGE_SIZE && packetsTotal == null;
}

function renderFlowsTable() {
  const flows = currentFlows();
  const extraction = mode === "live" ? (liveSnap && liveSnap.extraction) : (uploadStatus && uploadStatus.extraction);
  if (!flows) {
    const flowCount = extraction ? extraction.flow_count : (mode === "upload" && uploadStatus ? uploadStatus.flow_count : null);
    el("flows-tbody").innerHTML = flowCount
      ? `<tr class="empty-row"><td colspan="7">${flowCount} flow(s) extracted — per-flow detail available after State Prediction runs.</td></tr>`
      : '<tr class="empty-row"><td colspan="7">No flow data yet — run Feature Extraction / Prediction.</td></tr>';
    el("table-row-count").textContent = "0 rows";
    return;
  }
  const rows = applyTextFilters(flows,
    (f) => `${pickField(f, ["Protocol", "protocol"], "")}`,
    (f) => `${pickField(f, ["Src IP", "src_ip"], "")} ${pickField(f, ["Dst IP", "dst_ip"], "")}`);
  el("table-row-count").textContent = `${rows.length} rows`;
  el("flows-tbody").innerHTML = rows.map((f, i) => `
    <tr>
      <td>${i + 1}</td><td>${escapeHtml(pickField(f, ["Timestamp", "timestamp"], "N/A"))}</td>
      <td>${escapeHtml(pickField(f, ["Src IP", "src_ip"], "N/A"))}</td>
      <td>${escapeHtml(pickField(f, ["Src Port", "src_port"], "N/A"))}</td>
      <td>${escapeHtml(pickField(f, ["Dst IP", "dst_ip"], "N/A"))}</td>
      <td>${escapeHtml(pickField(f, ["Dst Port", "dst_port"], "N/A"))}</td>
      <td>${escapeHtml(pickField(f, ["Protocol", "protocol"], "N/A"))}</td>
    </tr>`).join("");
}

const FEATUREVALUE_COLS = [
  ["src_ip", "Src IP"], ["src_port", "Src Port"], ["dst_ip", "Dst IP"], ["dst_port", "Dst Port"],
  ["protocol", "Proto"],
  ["flow_duration", "Duration"], ["tot_fwd_pkts", "Fwd Pkts"], ["tot_bwd_pkts", "Bwd Pkts"],
  ["totlen_fwd_pkts", "Fwd Bytes"], ["totlen_bwd_pkts", "Bwd Bytes"],
  ["flow_pkts_s", "Pkts/s"], ["flow_byts_s", "Bytes/s"],
  ["flow_iat_mean", "IAT mean"], ["flow_iat_var", "IAT var"], ["flow_iat_max", "IAT max"],
  ["tcp_flags", "Flags"], ["tcp_flag_bitmask", "Flag mask"],
  ["syn_flag_cnt", "SYN"], ["ack_flag_cnt", "ACK"], ["fin_flag_cnt", "FIN"],
  ["rst_flag_cnt", "RST"], ["psh_flag_cnt", "PSH"], ["urg_flag_cnt", "URG"],
  ["down_up_ratio", "Down/Up"], ["init_fwd_win_byts", "Init Fwd Win"], ["init_bwd_win_byts", "Init Bwd Win"],
];

function renderFeatureValuesTable() {
  const flows = currentFlows();
  const head = el("featurevalues-head");
  const body = el("featurevalues-tbody");
  if (!flows) {
    head.innerHTML = "";
    body.innerHTML = '<tr class="empty-row"><td>No flow data yet — run State Prediction.</td></tr>';
    el("table-row-count").textContent = "0 rows";
    return;
  }
  const rows = applyTextFilters(flows,
    (f) => `${pickField(f, ["Protocol", "protocol"], "")}`,
    (f) => `${pickField(f, ["Src IP", "src_ip"], "")} ${pickField(f, ["Dst IP", "dst_ip"], "")}`);
  head.innerHTML = "<th>#</th>" + FEATUREVALUE_COLS.map(([, label]) => `<th>${escapeHtml(label)}</th>`).join("");
  el("table-row-count").textContent = `${rows.length} rows`;
  body.innerHTML = rows.map((f, i) => {
    const fv = f.features || {};
    const cells = FEATUREVALUE_COLS.map(([key]) => {
      let v = fv[key];
      if (v === undefined || v === null) v = pickField(f, [key, key.replace(/_/g, " ")], "");
      return `<td>${escapeHtml(String(v))}</td>`;
    }).join("");
    return `<tr><td>${i + 1}</td>${cells}</tr>`;
  }).join("");
}

function renderPredictionsTable() {
  const flows = currentFlows();
  if (!flows) {
    el("predictions-tbody").innerHTML = '<tr class="empty-row"><td colspan="6">No predictions yet — run State Prediction.</td></tr>';
    el("table-row-count").textContent = "0 rows";
    return;
  }
  const stateFilter = el("predictions-state-filter").value;
  let rows = applyTextFilters(flows,
    (f) => `${pickField(f, ["Protocol", "protocol"], "")}`,
    (f) => `${pickField(f, ["Src IP", "src_ip"], "")} ${pickField(f, ["Dst IP", "dst_ip"], "")}`);
  if (stateFilter) rows = rows.filter((f) => f.predicted_state === stateFilter);
  el("table-row-count").textContent = `${rows.length} rows`;
  el("predictions-tbody").innerHTML = rows.map((f, i) => `
    <tr>
      <td>${i + 1}</td>
      <td>${escapeHtml(pickField(f, ["Src IP", "src_ip"], "N/A"))}</td>
      <td>${escapeHtml(pickField(f, ["Dst IP", "dst_ip"], "N/A"))}</td>
      <td>${escapeHtml(pickField(f, ["Protocol", "protocol"], "N/A"))}</td>
      <td class="state-${escapeHtml(f.predicted_state)}">${escapeHtml(f.predicted_state)}</td>
      <td>${f.confidence}</td>
    </tr>`).join("");
}

function switchTemporalSubtab(tab) {
  activeTemporalSubtab = tab;
  document.querySelectorAll('#temporal-subtabs .subtab').forEach((b) => b.classList.toggle("subtab-active", b.dataset.temporalSubtab === tab));
  for (const t of ["states", "transitions", "sequences", "validation"]) {
    el(`temporal-subtab-${t}`).style.display = t === tab ? "" : "none";
  }
  renderTemporalTable();
}

function renderTemporalTable() {
  const r = temporalStatus.result;
  const vreport = validationStatus.report;
  el("table-row-count").textContent = r ? `${r.total_windows} windows` : "0 rows";

  const statesEl = el("temporal-subtab-states");
  if (!r) {
    statesEl.innerHTML = renderTemporalStatesTimeline() +
      `<div class="panel-notice">Temporal dataset not prepared yet. Use "Prepare Temporal Dataset" in the TEMPORAL DATASET detail tab.</div>`;
  } else {
    statesEl.innerHTML = renderTemporalStatesTimeline() + `
      <div class="panel-notice" style="text-align:left; padding:0;">
        Row-level window data is not exposed by <code>/api/temporal/status</code> (summary counts only) —
        full detail lives in <code>${escapeHtml(r.output_dir)}/temporal_states.csv</code>. Summary from the last real run:
        <br/><br/>
        <table class="data-table" style="width:auto;">
          <tbody>
            <tr><td>Time windows</td><td>${r.total_windows}</td></tr>
            <tr><td>State features</td><td>${r.state_features}</td></tr>
            <tr><td>Transitions</td><td>${r.transitions}</td></tr>
            <tr><td>Sequences</td><td>${r.total_sequences}</td></tr>
            <tr><td>Train / Val / Test windows</td><td>${r.train_windows} / ${r.validation_windows} / ${r.test_windows}</td></tr>
            <tr><td>Train / Val / Test sequences</td><td>${r.train_sequences} / ${r.validation_sequences} / ${r.test_sequences}</td></tr>
            <tr><td>BENIGN / DDoS / DoS / PortScan (flows)</td><td>${r.state_distribution.BENIGN} / ${r.state_distribution.DDoS} / ${r.state_distribution.DoS} / ${r.state_distribution.PortScan}</td></tr>
          </tbody>
        </table>
      </div>`;
  }

  const transEl = el("temporal-subtab-transitions");
  const transCheck = vreport && vreport.details && vreport.details.transitions;
  if (transCheck && transCheck.details && transCheck.details.from_to_counts) {
    const d = transCheck.details;
    transEl.innerHTML = `
      <div class="panel-notice" style="text-align:left; padding:0;">
        Total transitions: <b>${d.total_transitions}</b> &nbsp;|&nbsp; Status: <span class="v-status-${transCheck.status}">${transCheck.status}</span>
        <br/><br/>
        <table class="data-table" style="width:auto;">
          <thead><tr><th>From</th><th>To</th><th>Count</th></tr></thead>
          <tbody>
            ${d.from_to_counts.map((r2) => `<tr><td class="state-${escapeHtml(r2.current_state)}">${escapeHtml(r2.current_state)}</td><td class="state-${escapeHtml(r2.next_state)}">${escapeHtml(r2.next_state)}</td><td>${r2.count}</td></tr>`).join("")}
          </tbody>
        </table>
      </div>`;
  } else if (r) {
    transEl.innerHTML = `<div class="panel-notice">${r.transitions} transitions generated. Run "Validate Temporal Dataset" (VALIDATION tab) for the from&#8594;to breakdown.</div>`;
  } else {
    transEl.innerHTML = `<div class="panel-notice">Not available yet.</div>`;
  }

  const seqEl = el("temporal-subtab-sequences");
  const seqCheck = vreport && vreport.details && vreport.details.sequences;
  if (seqCheck && seqCheck.details && seqCheck.status !== "NOT_AVAILABLE") {
    const d = seqCheck.details;
    seqEl.innerHTML = `
      <div class="panel-notice" style="text-align:left; padding:0;">
        Status: <span class="v-status-${seqCheck.status}">${seqCheck.status}</span>
        <table class="data-table" style="width:auto; margin-top:8px;">
          <tbody>
            <tr><td>Sequence length (configured)</td><td>${na(d.sequence_length_configured)}</td></tr>
            <tr><td>Sequence length (actual)</td><td>${na(d.sequence_length_actual)}</td></tr>
            <tr><td>Total sequences</td><td>${d.total_sequences}</td></tr>
            <tr><td>Expected sequences (N - L)</td><td>${d.expected_sequences_N_minus_L}</td></tr>
            <tr><td>Count correct</td><td>${d.sequence_count_correct ? "YES" : "NO"}</td></tr>
            <tr><td>Target misaligned</td><td>${d.target_misaligned}</td></tr>
            <tr><td>Duplicate sequences</td><td>${d.duplicate_sequences}</td></tr>
          </tbody>
        </table>
      </div>`;
  } else if (r) {
    seqEl.innerHTML = `<div class="panel-notice">${r.total_sequences} sequences generated (length ${r.sequence_length}). Run "Validate Temporal Dataset" (VALIDATION tab) for detailed sequence checks.</div>`;
  } else {
    seqEl.innerHTML = `<div class="panel-notice">Not available yet.</div>`;
  }

  const valEl = el("temporal-subtab-validation");
  if (vreport) {
    valEl.innerHTML = `<table class="data-table validation-grid" style="width:auto;">
      <thead><tr><th>Check</th><th>Result</th></tr></thead>
      <tbody>${CHECK_ORDER.map(([key, label]) => {
        const c = vreport.details[key];
        return `<tr><td>${label}</td><td class="v-status-${c.status}">${statusIcon(c.status)} ${c.status}</td></tr>`;
      }).join("")}</tbody>
    </table>`;
  } else {
    valEl.innerHTML = `<div class="panel-notice">Not validated yet. Use "Validate Temporal Dataset" in the TEMPORAL DATASET &#8594; VALIDATION detail tab.</div>`;
  }
}

function renderActiveTable() {
  if (activeTableTab === "packets") renderPacketsTable();
  else if (activeTableTab === "flows") renderFlowsTable();
  else if (activeTableTab === "predictions") renderPredictionsTable();
  else if (activeTableTab === "featurevalues") renderFeatureValuesTable();
  else if (activeTableTab === "temporal") renderTemporalTable();
}

/* ==================== Detail tabs ==================== */
function switchDetailTab(tab) {
  activeDetailTab = tab;
  document.querySelectorAll(".detail-tab").forEach((b) => b.classList.toggle("detail-tab-active", b.dataset.detailTab === tab));
  document.querySelectorAll(".detail-pane").forEach((p) => { p.style.display = p.dataset.detailPane === tab ? "" : "none"; });
  if (tab === "xdr") loadXdrCampaign();
}

/* ==================== XDR / campaign prototype ==================== */
function showXdrError(message) {
  const box = el("xdr-error");
  box.textContent = message || "";
  box.style.display = message ? "block" : "none";
}

async function loadXdrCampaign() {
  showXdrError("");
  try {
    const sessionId = mode === "upload" ? uploadSessionId : (liveSnap && liveSnap.capture && liveSnap.capture.session_id);
    const query = sessionId ? `?session_id=${encodeURIComponent(sessionId)}` : "";
    const [ingest, graph, triage, deception, audit] = await Promise.all([
      responseApi(`/api/ingest/zeek${query}`), responseApi(`/api/graph${query}`),
      responseApi(`/api/triage${query}`, { method: "POST" }),
      responseApi("/api/deception/hits"), responseApi("/api/response/audit"),
    ]);
    xdrEnrichment = ingest.enrichment || null;
    xdrGraph = graph;
    xdrTriage = triage;
    xdrHits = deception.hits || [];
    xdrAudit = audit.events || [];
    xdrPlan = [...xdrAudit].reverse().find((event) => event.event === "PLAN") || null;
    xdrAction = xdrPlan ? ([...xdrAudit].reverse().find((event) =>
      event.action_id === xdrPlan.action_id && ["APPLY", "ROLLBACK"].includes(event.event)) || null) : null;
    el("xdr-disabled").style.display = "none";
  } catch (error) {
    el("xdr-disabled").style.display = "block";
    showXdrError(error.status === 404 ? "XDR prototype is disabled or has no session data." : error.message);
  }
  renderXdrCampaign();
}

function renderXdrGraph(graph) {
  if (!graph || !(graph.nodes || []).length) return '<div class="dist-empty">No graph data.</div>';
  const nodes = graph.nodes.slice(0, 24);
  const ids = new Set(nodes.map((node) => node.id));
  const cx = 170, cy = 100, radius = 76;
  const positions = {};
  nodes.forEach((node, index) => {
    const angle = (Math.PI * 2 * index / nodes.length) - Math.PI / 2;
    positions[node.id] = [cx + radius * Math.cos(angle), cy + radius * Math.sin(angle)];
  });
  const surprising = new Set((graph.surprising_edges || []).map((edge) => `${edge.source}|${edge.target}`));
  const lines = (graph.edges || []).filter((edge) => ids.has(edge.source) && ids.has(edge.target)).map((edge) => {
    const a = positions[edge.source], b = positions[edge.target];
    const red = surprising.has(`${edge.source}|${edge.target}`);
    return `<line x1="${a[0]}" y1="${a[1]}" x2="${b[0]}" y2="${b[1]}" class="${red ? "xdr-edge-surprise" : "xdr-edge"}"><title>${escapeHtml(edge.source)} → ${escapeHtml(edge.target)} · surprise ${Number(edge.edge_surprise || 0).toFixed(2)}</title></line>`;
  }).join("");
  const circles = nodes.map((node) => {
    const point = positions[node.id];
    return `<g><circle cx="${point[0]}" cy="${point[1]}" r="5" class="xdr-node"><title>${escapeHtml(node.id)}</title></circle><text x="${point[0] + 7}" y="${point[1] + 3}">${escapeHtml(node.id)}</text></g>`;
  }).join("");
  return `<svg viewBox="0 0 340 200" role="img" aria-label="Session communication graph">${lines}${circles}</svg>`;
}

function renderXdrCampaign() {
  const sensorMap = [
    ["DNS entropy", "dns_query_entropy_mean", 5], ["Unique SNI", "unique_sni_count", 20],
    ["Beacon", "beacon_score_max", 1], ["Byte asymmetry", "byte_asymmetry_max", 1],
    ["JA3 novelty", "ja3_novelty", 1], ["HTTP errors", "http_error_rate", 1],
  ];
  el("xdr-sensors").innerHTML = xdrEnrichment ? sensorMap.map(([label, key, max]) => {
    const value = Number(xdrEnrichment[key] || 0);
    return `<div class="xdr-sensor"><span>${label}</span><div><i style="width:${Math.min(100, value / max * 100)}%"></i></div><strong>${value.toFixed(key === "unique_sni_count" ? 0 : 3)}</strong></div>`;
  }).join("") : '<div class="dist-empty">No enrichment data.</div>';
  el("xdr-campaign-score").textContent = xdrGraph ? Number(xdrGraph.campaign_score || 0).toFixed(2) : "N/A";
  el("xdr-graph").innerHTML = renderXdrGraph(xdrGraph);
  el("xdr-triage-summary").textContent = xdrTriage ? `${xdrTriage.summary} Confidence: ${xdrTriage.confidence}.` : "No triage data.";
  el("xdr-techniques").innerHTML = xdrTriage && xdrTriage.ranked_techniques.length ? xdrTriage.ranked_techniques.map((item) =>
    `<div><strong>${escapeHtml(item.technique_id)}</strong> ${escapeHtml(item.technique_name)} — ${Math.round(Number(item.confidence || 0) * 100)}%</div>`).join("") : '<div class="dist-empty">No ranked techniques.</div>';
  el("xdr-playbook").innerHTML = xdrTriage ? xdrTriage.playbook.map((step) => `<li>${escapeHtml(step)}</li>`).join("") : "";
  el("xdr-response-step").textContent = xdrPlan ? xdrPlan.step : "NONE";
  el("xdr-response-command").textContent = xdrPlan ? `${xdrPlan.command}\n\n${xdrPlan.ruleset || ""}` : "No response plan.";
  const ack = el("xdr-operator-ack").checked;
  el("btn-xdr-apply").disabled = !(xdrPlan && xdrPlan.step !== "NONE" && !xdrAction && (!xdrPlan.requires_operator_ack || ack));
  el("btn-xdr-rollback").disabled = !(xdrAction && xdrAction.status !== "DRY_RUN_ROLLED_BACK" && ack);
  el("xdr-audit").innerHTML = xdrAudit.length ? xdrAudit.slice(-12).reverse().map((event) => `<tr><td>${escapeHtml(event.event)}</td><td>${escapeHtml(event.step || "N/A")}</td><td>${escapeHtml(event.status || "PLANNED")}</td></tr>`).join("") : '<tr class="empty-row"><td colspan="3">No audit entries.</td></tr>';
  el("xdr-hits").innerHTML = xdrHits.length ? xdrHits.slice().reverse().map((hit) => `<tr><td>${escapeHtml(hit.timestamp)}</td><td>${escapeHtml(hit.source_ip)}</td><td>${escapeHtml(hit.user_agent)}</td></tr>`).join("") : '<tr class="empty-row"><td colspan="3">No honeytoken hits.</td></tr>';
}

async function applyXdrDryRun() {
  if (!xdrPlan) return;
  try {
    xdrAction = await responseApi("/api/response/apply", { method: "POST", body: JSON.stringify({ plan_id: xdrPlan.plan_id, operator_ack: el("xdr-operator-ack").checked }) });
    await loadXdrCampaign();
  } catch (error) { showXdrError(error.message); }
}

async function rollbackXdrDryRun() {
  if (!xdrAction) return;
  try {
    await responseApi("/api/response/rollback", { method: "POST", body: JSON.stringify({ action_id: xdrAction.action_id, operator_ack: el("xdr-operator-ack").checked }) });
    await loadXdrCampaign();
  } catch (error) { showXdrError(error.message); }
}

/* ==================== Controlled firewall response ==================== */
function responsePrediction() {
  return mode === "live" ? (liveSnap && liveSnap.prediction) : (uploadStatus && uploadStatus.prediction);
}

function responseReference() {
  return mode === "live" ? { mode: "live" } : { mode: "upload", session_id: uploadSessionId };
}

async function responseApi(path, opts = {}) {
  const headers = { ...(opts.headers || {}) };
  if (opts.body) headers["Content-Type"] = "application/json";
  if (responseToken) headers["X-NIDS-Response-Token"] = responseToken;
  return api(path, { ...opts, headers });
}

function showResponseError(message) {
  const box = el("response-error");
  box.textContent = message || "";
  box.style.display = message ? "block" : "none";
}

async function loadResponseBootstrap() {
  try {
    responseCapabilities = await api("/api/response/capabilities");
    responseToken = responseCapabilities.local_authorization_token;
    await loadResponseHistory();
    showResponseError("");
  } catch (error) {
    showResponseError("Response subsystem unavailable: " + error.message);
  }
  renderResponseTab();
}

async function loadResponseHistory() {
  const result = await responseApi("/api/response/actions");
  responseHistory = result.actions || [];
}

async function inspectResponseAction(actionId) {
  showResponseError("");
  try {
    responseAction = await responseApi(`/api/response/actions/${encodeURIComponent(actionId)}`);
    responseAudit = responseAction.events || [];
  } catch (error) { showResponseError(error.message); }
  renderResponseTab();
}

async function scanFirewall() {
  showResponseError("");
  el("btn-response-scan").disabled = true;
  try {
    responseScan = await responseApi("/api/response/scan", { method: "POST" });
  } catch (error) {
    showResponseError(error.message);
  }
  el("btn-response-scan").disabled = false;
  renderResponseTab();
}

async function createResponsePlan() {
  showResponseError("");
  try {
    responsePlan = await responseApi("/api/response/plans", {
      method: "POST",
      body: JSON.stringify({
        prediction_reference: responseReference(),
        ttl_minutes: Number(el("response-ttl").value),
      }),
    });
    responseAction = null;
    responseAudit = [];
    el("response-apply-ack").checked = false;
    el("response-rollback-ack").checked = false;
    await loadResponseHistory();
  } catch (error) {
    showResponseError(error.message);
  }
  renderResponseTab();
}

async function applyResponsePlan() {
  if (!responsePlan || !el("response-apply-ack").checked) return;
  showResponseError("");
  try {
    responseAction = await responseApi(`/api/response/plans/${encodeURIComponent(responsePlan.plan_id)}/apply`, {
      method: "POST",
      body: JSON.stringify({ plan_hash: responsePlan.plan_hash, confirmed: true }),
    });
    responseAudit = responseAction.events || [];
    await loadResponseHistory();
  } catch (error) {
    showResponseError(error.message);
  }
  renderResponseTab();
}

async function verifyResponseAction() {
  if (!responseAction) return;
  showResponseError("");
  try {
    responseAction = await responseApi(`/api/response/actions/${encodeURIComponent(responseAction.action_id)}/verify`, {
      method: "POST", body: JSON.stringify({ confirmed: true }),
    });
    responseAudit = responseAction.events || [];
    await loadResponseHistory();
  } catch (error) { showResponseError(error.message); }
  renderResponseTab();
}

async function rollbackResponseAction() {
  if (!responseAction || !el("response-rollback-ack").checked) return;
  showResponseError("");
  try {
    responseAction = await responseApi(`/api/response/actions/${encodeURIComponent(responseAction.action_id)}/rollback`, {
      method: "POST", body: JSON.stringify({ confirmed: true }),
    });
    responseAudit = responseAction.events || [];
    await loadResponseHistory();
  } catch (error) { showResponseError(error.message); }
  renderResponseTab();
}

function renderResponseTab() {
  const caps = responseCapabilities;
  const prediction = responsePrediction();
  const actions = (caps && caps.supported_actions) || [];
  el("response-engine").textContent = caps ? `${caps.platform} / ${caps.engine}` : "N/A";
  el("response-privilege").textContent = caps ? (caps.privilege_ready ? "READY" : "HELPER NOT CONFIGURED") : "N/A";
  el("response-health").textContent = responseScan ? (responseScan.namespace_healthy ? "HEALTHY" : "UNAVAILABLE") : "NOT SCANNED";
  el("response-conflicts").textContent = responseScan && responseScan.conflicts.length ? responseScan.conflicts.join(", ") : "NONE OBSERVED";
  el("btn-response-plan").disabled = !prediction || !responseToken;

  const eligibility = responsePlan && responsePlan.eligibility;
  el("response-source").textContent = eligibility ? eligibility.confidence_source : "N/A";
  el("response-attack").textContent = eligibility ? (eligibility.attack_class || "N/A") : "N/A";
  el("response-eligibility").textContent = !eligibility ? "NO PLAN" : (eligibility.executable ? "EXECUTABLE PROPOSAL" : "RECOMMENDATION ONLY");
  const notes = responsePlan ? [
    ...(responsePlan.warnings || []), ...(responsePlan.limitations || []),
    ...(responsePlan.upstream_recommendation ? [responsePlan.upstream_recommendation] : []),
  ] : [];
  el("response-limitations").textContent = notes.join(" ") || "Run a current prediction to request a response recommendation.";
  el("response-plan-hash").textContent = responsePlan ? responsePlan.plan_hash : "N/A";
  const preview = responsePlan ? {
    targets: responsePlan.targets,
    ttl_minutes: responsePlan.ttl_minutes,
    exact_native_changes: responsePlan.native_changes.commands,
    affected_traffic: responsePlan.native_changes.affected_traffic,
    rollback_effect: responsePlan.native_changes.rollback,
    warnings: responsePlan.warnings,
  } : "No plan generated.";
  el("response-preview").textContent = typeof preview === "string" ? preview : JSON.stringify(preview, null, 2);

  const applyReady = Boolean(eligibility && eligibility.executable && actions.includes("apply") && el("response-apply-ack").checked);
  el("btn-response-apply").disabled = !applyReady;
  const state = responseAction ? responseAction.state : "N/A";
  el("response-action-state").textContent = state;
  el("response-action-state").className = state === "N/A" ? "" : `response-state-${state}`;
  el("response-native-ids").textContent = responseAction && responseAction.native_identifiers.length ? responseAction.native_identifiers.join(", ") : "N/A";
  el("btn-response-verify").disabled = !(responseAction && ["APPLIED", "VERIFY_FAILED"].includes(responseAction.state) && actions.includes("verify"));
  el("btn-response-rollback").disabled = !(responseAction && responseAction.state === "VERIFIED" && actions.includes("rollback") && el("response-rollback-ack").checked);

  const filter = el("response-history-filter").value;
  const filtered = responseHistory.filter((action) => !filter || action.state === filter);
  el("response-history").innerHTML = filtered.length ? filtered.map((action) => `<tr>
    <td>${escapeHtml(new Date(action.created_at).toLocaleString())}</td>
    <td class="response-state-${escapeHtml(action.state)}">${escapeHtml(action.state)}</td>
    <td>${escapeHtml(action.actor || "N/A")}</td>
    <td><button class="btn btn-sm response-inspect" data-action-id="${escapeHtml(action.action_id)}" title="${escapeHtml(action.action_id)}">${escapeHtml(action.action_id.slice(0, 8))}</button></td>
  </tr>`).join("") : '<tr class="empty-row"><td colspan="4">No matching response actions.</td></tr>';
  document.querySelectorAll(".response-inspect").forEach((button) => button.addEventListener("click", () => inspectResponseAction(button.dataset.actionId)));
  el("response-event-action").textContent = responseAction ? responseAction.action_id.slice(0, 8) : "select an action";
  const eventFilter = el("response-event-filter").value;
  const events = responseAudit.filter((event) => !eventFilter || event.state === eventFilter);
  el("response-events").innerHTML = events.length ? events.map((event) => `<tr>
    <td>${escapeHtml(new Date(event.created_at).toLocaleString())}</td>
    <td class="response-state-${escapeHtml(event.state)}">${escapeHtml(event.state)}</td>
    <td>${escapeHtml(event.actor || "N/A")}</td>
  </tr>`).join("") : '<tr class="empty-row"><td colspan="3">No matching events.</td></tr>';
}

const STOP_REASON_LABEL = {
  duration_target_reached: "Capture complete — duration target reached",
  packet_target_reached: "Capture complete — packet target reached",
  target_reached: "Capture complete — target reached",
  user_stopped: "Capture stopped by user",
  safety_timeout: "Capture stopped — safety timeout reached",
};

function fmtCapturePackets(cap) {
  // While capturing, show real progress toward the packet target — only
  // set at all when a caller explicitly opted into one; the default
  // capture has no packet_target (duration is the primary target, shown
  // separately by fmtCaptureDuration), so this just shows the live count.
  if (!cap || cap.packet_count == null) return cap && cap.status === "CAPTURING" ? "…" : "N/A";
  if (cap.status === "CAPTURING" && cap.packet_target) return `${cap.packet_count} / ${cap.packet_target}`;
  return String(cap.packet_count);
}

function fmtCaptureDuration(cap) {
  // Duration is the PRIMARY capture target (dumpcap's own -a duration:N),
  // so while capturing this shows real progress toward it, the same way
  // fmtCapturePackets shows progress toward an (optional) packet target.
  if (!cap) return "--";
  const elapsed = fmtDuration(cap.duration_seconds);
  if (cap.status === "CAPTURING" && cap.duration_target) return `${elapsed} / ${cap.duration_target}s`;
  return elapsed;
}

function renderCaptureTab() {
  if (mode === "live") {
    const cap = liveSnap ? liveSnap.capture : null;
    const capturing = cap && cap.status === "CAPTURING";

    // B3 — feed the packets/sec sparkline from this same poll.
    if (capturing && cap && cap.packet_count != null) {
      const last = packetRateHistory[packetRateHistory.length - 1];
      if (!last || cap.packet_count < last.packets) packetRateHistory = [];  // new capture / counter reset
      if (!last || cap.packet_count !== last.packets || Date.now() - last.t > 900) {
        packetRateHistory.push({ t: Date.now(), packets: cap.packet_count });
        if (packetRateHistory.length > 40) packetRateHistory.shift();
      }
    } else if (!capturing && cap && cap.status !== "STOPPED") {
      packetRateHistory = [];
    }
    renderCaptureRateSparkline();
    el("btn-start").disabled = capturing;
    el("btn-stop").disabled = !capturing;
    el("iface-select").disabled = capturing;
    el("capture-dot").className = "status-dot " + (
      liveSnap && liveSnap.stage === "ERROR" ? "status-error" : capturing ? "status-capturing" : cap ? "status-stopped" : "status-idle");
    el("capture-status-text").textContent = (liveSnap && liveSnap.stage === "ERROR") ? "ERROR" : (cap ? cap.status : "IDLE");
    el("metric-duration").textContent = cap ? fmtCaptureDuration(cap) : "--";
    el("metric-packets").textContent = cap ? fmtCapturePackets(cap) : "--";

    el("d-interface").textContent = na(cap && cap.interface);
    el("d-capture-status").textContent = cap ? cap.status : "IDLE";
    el("d-start-time").textContent = cap ? fmtTime(cap.start_time) : "N/A";
    el("d-duration").textContent = cap ? fmtDuration(cap.duration_seconds) + "s" : "N/A";
    el("d-packet-count").textContent = cap ? fmtCapturePackets(cap) : "N/A";
    el("d-packet-target").textContent = cap
      ? [
          cap.duration_target ? `${cap.duration_target}s duration` : "unbounded duration",
          cap.packet_target ? `${cap.packet_target} packets` : null,
        ].filter(Boolean).join(" + ")
      : "N/A";
    el("d-pcap-file").textContent = na(cap && cap.pcap_path);

    const dropsRow = el("d-drops-row");
    if (cap && (cap.packets_received != null || cap.packets_dropped != null)) {
      dropsRow.style.display = "";
      const rec = cap.packets_received != null ? cap.packets_received : "?";
      const drp = cap.packets_dropped != null ? cap.packets_dropped : "?";
      const dEl = el("d-drops");
      dEl.textContent = `${rec} / ${drp}`;
      dEl.style.color = (cap.packets_dropped > 0) ? "var(--amber)" : "";
    } else {
      dropsRow.style.display = "none";
    }

    const stopRow = el("d-stop-reason-row");
    if (cap && cap.status === "STOPPED" && cap.stop_reason) {
      stopRow.style.display = "";
      el("d-stop-reason").textContent = STOP_REASON_LABEL[cap.stop_reason] || cap.stop_reason;
    } else {
      stopRow.style.display = "none";
    }

    el("d-hero-dot").className = "status-dot " + (capturing ? "status-capturing" : cap ? "status-stopped" : "status-idle");
    el("d-hero-status").textContent = cap ? cap.status : "IDLE";
    el("d-hero-packets").textContent = cap ? fmtCapturePackets(cap) : "N/A";
    el("d-hero-duration").textContent = cap ? fmtDuration(cap.duration_seconds) : "N/A";

    if (!capturing) {
      stopPacketPolling();
      if (wasCapturing) {
        // Capture just ended (either auto-stopped at the target or the
        // user pressed Stop) — the polling timer that was keeping the
        // packet table/pagination fresh has now stopped, so fetch once
        // more explicitly. Without this, the pagination bar would keep
        // showing the last mid-capture total ("of ?") until the user
        // manually changes page, even though the real, final total is
        // already known server-side.
        pollPackets();
      }
    }
    wasCapturing = capturing;
  } else {
    const u = uploadStatus;
    el("d-interface").textContent = "N/A (file upload)";
    el("d-capture-status").textContent = u ? u.stage : "IDLE";
    el("d-start-time").textContent = "N/A";
    el("d-duration").textContent = u && u.processing_seconds != null ? u.processing_seconds + "s (total processing)" : "N/A";
    el("d-packet-count").textContent = u && u.packet_count != null ? u.packet_count : "N/A";
    el("d-pcap-file").textContent = u ? `${na(u.filename)} (${u.input_type ? u.input_type.toUpperCase() : "?"}, ${fmtBytes(u.file_size)})` : "N/A";

    el("d-hero-dot").className = "status-dot " + (u && u.stage === "ERROR" ? "status-error" : u && u.stage === "PREDICTION_COMPLETED" ? "status-stopped" : u ? "status-capturing" : "status-idle");
    el("d-hero-status").textContent = u ? u.stage : "IDLE";
    el("d-hero-packets").textContent = u && u.packet_count != null ? u.packet_count : (u && u.row_count != null ? `${u.row_count} rows` : "N/A");
    el("d-hero-duration").textContent = u && u.processing_seconds != null ? u.processing_seconds + "s" : "N/A";
  }
}

let featureSpaceRendered = false;
function renderFeatureSpace(usedFeatureSet) {
  const body = el("feature-space-body");
  let html = "";
  for (const [group, names] of Object.entries(FEATURE_GROUPS)) {
    html += `<div class="feature-group"><div class="feature-group-title">${group}</div><ul class="feature-list">`;
    for (const name of names) {
      const isIdOnly = ID_ONLY_FIELDS.has(name);
      let statusHtml;
      if (isIdOnly) {
        statusHtml = `<span class="feature-status">&mdash; id/metadata</span>`; // never fed to the ANN, regardless of prediction status
      } else {
        const used = usedFeatureSet === "all"; // true only once prediction has actually succeeded (see caller)
        statusHtml = `<span class="feature-status ${used ? "fs-used" : ""}">${used ? "&#10003; used in inference" : "&mdash;"}</span>`;
      }
      html += `<li><span>${escapeHtml(name)}</span>${statusHtml}</li>`;
    }
    html += `</ul></div>`;
  }
  body.innerHTML = html;
}

function renderExtractionTab() {
  const ext = mode === "live" ? (liveSnap && liveSnap.extraction) : (uploadStatus && uploadStatus.extraction);
  const stage = mode === "live" ? (liveSnap && liveSnap.stage) : (uploadStatus && uploadStatus.stage);
  const cap = mode === "live" ? (liveSnap && liveSnap.capture) : null;

  if (mode === "upload" && uploadStatus && uploadStatus.input_type === "csv") {
    el("e-input").textContent = "N/A — CSV uploaded directly (no extraction step)";
    el("e-output").textContent = "N/A";
    el("e-status").textContent = "SKIPPED (CSV input)";
    el("e-flows").textContent = uploadStatus.flow_count != null ? uploadStatus.flow_count : "N/A";
    el("e-features").textContent = "N/A";
    el("e-time").textContent = "N/A";
    el("btn-extract").style.display = "none";
    el("btn-extract").disabled = true;
  } else {
    const canExtract = mode === "live" && cap && cap.status === "STOPPED" && stage !== "EXTRACTING" && stage !== "PREDICTING";
    el("btn-extract").style.display = mode === "live" ? "" : "none";
    el("btn-extract").disabled = !canExtract;
    el("e-input").textContent = na(ext && ext.input_pcap);
    el("e-output").textContent = na(ext && ext.output_csv);
    el("e-status").textContent = stage === "EXTRACTING" ? "RUNNING" : (ext ? "COMPLETED" : (stage === "ERROR" && !ext ? "FAILED" : "IDLE"));
    el("e-flows").textContent = ext ? ext.flow_count : "N/A";
    el("e-features").textContent = ext ? ext.feature_count : "N/A";
    el("e-time").textContent = ext ? ext.extraction_seconds + "s" : "N/A";
  }

  const pred = mode === "live" ? (liveSnap && liveSnap.prediction) : (uploadStatus && uploadStatus.prediction);
  renderFeatureSpace(pred ? "all" : "none");
}

let packetFeatureData = null;
async function loadPacketFeatures() {
  const btn = el("btn-packet-features");
  if (btn) { btn.disabled = true; btn.textContent = "Computing…"; }
  try {
    const body = {};
    if (mode === "upload" && uploadStatus && uploadStatus.session_id) body.upload_session_id = uploadStatus.session_id;
    packetFeatureData = await api("/api/extract/packet", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
    });
    renderPacketFeatures();
  } catch (e) {
    el("packet-features-body").innerHTML = `<div class="dist-empty">${escapeHtml(e.message || "failed")}</div>`;
  } finally {
    if (btn) { btn.disabled = false; btn.textContent = "Recompute packet-level features"; }
  }
}

function renderPacketFeatures() {
  const d = packetFeatureData;
  const card = el("port-scan-card");
  const body = el("packet-features-body");
  if (!d) return;
  const ps = d.port_scan || {};
  if (ps.pattern && ps.pattern !== "none") {
    card.style.display = "";
    card.className = "port-scan-card port-scan-hit";
    card.textContent = `PORT SCAN: ${ps.pattern} sweep ${ps.src || "?"} → ${ps.dst || "?"}, ` +
      `${ps.unique_ports} ports` + (ps.port_range ? ` (${ps.port_range[0]}–${ps.port_range[1]})` : "");
  } else {
    card.style.display = "";
    card.className = "port-scan-card";
    card.textContent = "PORT SCAN: no sequential/randomised port sweep detected";
  }
  const cols = d.columns || [];
  const rows = (d.flows || []).slice(0, 500);
  body.innerHTML = `<table class="data-table"><thead><tr>${
    cols.map(c => `<th>${escapeHtml(c)}</th>`).join("")
  }</tr></thead><tbody>${
    rows.map(r => `<tr>${cols.map(c => `<td>${escapeHtml(String(r[c] ?? ""))}</td>`).join("")}</tr>`).join("")
  }</tbody></table><div class="dist-empty" style="text-align:left">${d.flow_count} bidirectional flow(s)${rows.length < (d.flows || []).length ? " (first 500 shown)" : ""}</div>`;
}

let classifierMetricsData = null;
async function loadClassifierMetrics() {
  const btn = el("btn-classifier-metrics");
  if (btn) { btn.disabled = true; btn.textContent = "Computing…"; }
  try {
    classifierMetricsData = await api("/api/predict/metrics");
    renderClassifierMetrics();
  } catch (e) {
    el("classifier-metrics-body").innerHTML = `<div class="dist-empty">${escapeHtml(e.message || "failed")}</div>`;
    el("classifier-metrics-source").style.display = "none";
  } finally {
    if (btn) { btn.disabled = false; btn.textContent = "Recompute classifier metrics"; }
  }
}

function renderClassifierMetrics() {
  const m = classifierMetricsData;
  const src = el("classifier-metrics-source");
  const body = el("classifier-metrics-body");
  if (!m) return;
  const gt = m.is_ground_truth;
  src.style.display = "";
  src.className = "metrics-source " + (gt ? "metrics-source-gt" : "metrics-source-proxy");
  src.textContent = gt
    ? `Ground truth · ${m.source} · n=${m.n}`
    : `PROXY (not ground truth) · ${m.source} · n=${m.n} · agreement ${((m.agreement_rate || 0) * 100).toFixed(1)}%`;
  const pc = m.per_class || {};
  const f = (x) => (x == null || typeof x !== "number") ? "—" : x.toFixed(3);
  const rows = (m.labels || Object.keys(pc)).map((c) => {
    const v = pc[c] || {};
    return `<tr><td>${escapeHtml(c)}</td><td>${f(v.precision)}</td><td>${f(v.recall)}</td><td>${f(v.f1)}</td><td>${v.support ?? 0}</td></tr>`;
  }).join("");
  const a = m.attack || {};
  const cm = m.confusion_matrix || [];
  const maxCell = Math.max(1, ...cm.flat());
  const cmHtml = cm.length ? `
    <div class="detail-col-subtitle" style="margin-top:8px;">Confusion — rows = ${gt ? "true" : "signature"}, cols = ANN predicted</div>
    <table class="data-table confusion-grid"><thead><tr><th></th>${(m.labels || []).map(c => `<th>${escapeHtml(c)}</th>`).join("")}</tr></thead>
    <tbody>${cm.map((row, i) => `<tr><th>${escapeHtml((m.labels || [])[i] || i)}</th>${row.map((n, j) => {
      const bg = `rgba(90,150,255,${(n / maxCell).toFixed(3)})`;
      return `<td style="background:${n ? bg : "transparent"}${i === j ? ";font-weight:700" : ""}">${n}</td>`;
    }).join("")}</tr>`).join("")}</tbody></table>` : "";
  body.innerHTML = `
    <table class="data-table" style="width:100%; margin-top:6px;">
      <thead><tr><th>Class</th><th>Prec.</th><th>Recall</th><th>F1</th><th>Support</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>
    <div class="dist-row" style="margin-top:6px;"><span class="dist-label">Macro F1</span><span class="dist-pct">${f(m.macro_f1)}</span></div>
    <div class="dist-row"><span class="dist-label">Weighted F1</span><span class="dist-pct">${f(m.weighted_f1)}</span></div>
    <div class="dist-row"><span class="dist-label">Accuracy</span><span class="dist-pct">${f(m.accuracy)}</span></div>
    <div class="dist-row"><span class="dist-label">Attack precision</span><span class="dist-pct">${f(a.precision)}</span></div>
    <div class="dist-row"><span class="dist-label">Attack recall</span><span class="dist-pct">${f(a.recall)}</span></div>
    <div class="dist-row"><span class="dist-label">Attack F1</span><span class="dist-pct">${f(a.f1)}</span></div>
    <div class="dist-row"><span class="dist-label">Attack false-positive rate</span><span class="dist-pct">${f(a.false_positive_rate)}</span></div>
    ${cmHtml}
    ${m.note ? `<div class="dist-empty" style="text-align:left; margin-top:6px;">${escapeHtml(m.note)}</div>` : ""}`;
}

/* Shared display verdict — honours the confidence gate (predict.py
   verdict_gate min_ratio: the attack class's own % share of scored flows).
   A small fraction of ANN-only flagged flows in a mostly-benign capture
   reads BENIGN/SUSPICIOUS, never ATTACK, unless the signature layer
   confirmed it (confidence "high"). */
function predVerdict(pred) {
  if (!pred) return { tier: "none", cls: null };
  const cls = (pred.attack_alert && pred.attack_alert !== "NONE" ? pred.attack_alert : null)
    || pred.effective_attack_class || pred.attack_class || null;
  const conf = pred.effective_confidence || pred.confidence;
  if (!cls) return { tier: "benign", cls: null };
  if (conf === "high") return { tier: "attack", cls };
  return { tier: "suspicious", cls };
}

function renderPredictionTab() {
  const pred = mode === "live" ? (liveSnap && liveSnap.prediction) : (uploadStatus && uploadStatus.prediction);
  const stage = mode === "live" ? (liveSnap && liveSnap.stage) : (uploadStatus && uploadStatus.stage);
  const ext = mode === "live" ? (liveSnap && liveSnap.extraction) : null;
  const canPredictLive = mode === "live" && ext && stage !== "PREDICTING" && stage !== "EXTRACTING";
  el("btn-predict").style.display = mode === "live" ? "" : "none";
  el("btn-predict").disabled = !canPredictLive;

  const stateVal = el("p-state-value");
  // Effective = ANN verdict, escalated where the signature layer disagrees.
  const v = predVerdict(pred);
  if (!pred) {
    stateVal.textContent = "N/A";
    stateVal.className = "";
  } else if (v.tier === "attack") {
    stateVal.textContent = "ATTACK: " + v.cls;
    stateVal.className = "state-" + v.cls + " verdict-attack";
  } else if (v.tier === "suspicious") {
    // low confidence — a few ANN-only flows, no deterministic confirmation
    stateVal.textContent = "SUSPICIOUS: " + v.cls + "?";
    stateVal.className = "state-" + v.cls + " verdict-suspicious";
  } else {
    stateVal.textContent = "BENIGN";
    stateVal.className = "state-BENIGN";
  }

  const domEl = el("p-dominant");
  if (domEl) {
    if (!pred) {
      domEl.textContent = "";
    } else {
      let sub = `Dominant traffic: ${pred.dominant_state} · attack alert: ${pred.attack_alert || "NONE"}`;
      if (pred.attack_class) {
        sub += ` · ANN: ${pred.attack_class_counts[pred.attack_class]} ${pred.attack_class} flow(s)`;
        if (typeof pred.attack_class_ratio === "number") {
          sub += ` (${(pred.attack_class_ratio * 100).toFixed(1)}%)`;
        }
      }
      if (pred.signature_verdict && pred.signature_verdict !== pred.attack_class) {
        sub += ` · signature: ${pred.signature_verdict}`;
      }
      domEl.textContent = sub;
    }
  }

  const flagged = el("p-flagged");
  if (flagged) {
    if (pred && pred.flows_analyzed) {
      const pct = ((pred.attack_ratio ?? pred.malicious_flow_ratio ?? 0) * 100).toFixed(1);
      flagged.textContent = `${pred.attack_flow_count} of ${pred.flows_analyzed} flows flagged (${pct}%)`;
      flagged.style.color = v.tier === "attack" ? "var(--red)" : v.tier === "suspicious" ? "var(--amber)" : "var(--green)";
    } else {
      flagged.textContent = "";
    }
  }

  el("p-model").textContent = pred ? pred.model : "N/A";
  el("p-flows").textContent = pred ? pred.flows_analyzed : "N/A";
  el("p-status").textContent = stage === "PREDICTING" ? "RUNNING" : (pred ? "COMPLETED" : (stage === "ERROR" && !pred ? "FAILED" : "IDLE"));
  el("p-time").textContent = pred ? pred.inference_seconds + "s" : "N/A";
  el("p-output-csv").textContent = pred ? na(pred.output_csv) : "N/A";

  const btnDl = el("btn-download-csv");
  btnDl.disabled = !(mode === "upload" && uploadStatus && uploadStatus.stage === "PREDICTION_COMPLETED");

  renderClassDistribution(pred);
  renderSignatureHits(pred);
  renderDrivingFeatures(pred);
  renderPredictionFlowsTimeline(pred);
}

function renderSignatureHits(pred) {
  const wrap = el("signature-hits");
  if (!wrap) return;
  if (!pred) {
    wrap.innerHTML = '<div class="dist-empty">N/A — no prediction yet</div>';
    return;
  }
  const hits = pred.signature_hits || [];
  if (!hits.length) {
    wrap.innerHTML = '<div class="dist-empty">No signature rule fired</div>';
    return;
  }
  wrap.innerHTML = hits.map(h => `<div class="dist-row">
    <span class="dist-label state-${escapeHtml(h.state)}" title="${escapeHtml(h.rule)}">${escapeHtml(h.state)}</span>
    <span class="sig-detail">${escapeHtml(h.detail)}</span>
    <span class="dist-pct">${h.flow_count} flow${h.flow_count === 1 ? "" : "s"}</span>
  </div>`).join("");
}

function renderDrivingFeatures(pred) {
  const wrap = el("driving-features");
  if (!wrap) return;
  const job = pred && pred.explanation_jobs && pred.explanation_jobs[0];
  const shapResult = job && explanationResults[job.job_id];
  if (job && !shapResult && !explanationPolling.has(job.job_id)) pollExplanation(job.job_id);
  const feats = shapResult
    ? shapResult.feature_contributions.map(item => ({
        feature: item.feature,
        mean_abs_contribution: Math.abs(item.contribution),
        mean_signed_contribution: item.contribution,
      }))
    : pred && pred.driving_features;
  const method = el("attribution-method");
  if (method) method.textContent = shapResult
    ? `${shapResult.method} · class ${shapResult.explained_class} · base ${shapResult.base_value == null ? "N/A" : Number(shapResult.base_value).toFixed(4)}`
    : (pred ? "gradient × input fallback — not SHAP" : "awaiting asynchronous SHAP");
  if (!feats || !feats.length) {
    wrap.innerHTML = '<div class="dist-empty">N/A — no prediction yet</div>';
    return;
  }
  const max = Math.max(...feats.map(f => Math.abs(f.mean_abs_contribution))) || 1;
  wrap.innerHTML = feats.map(f => {
    const pct = Math.round((Math.abs(f.mean_abs_contribution) / max) * 100);
    const color = f.mean_signed_contribution >= 0 ? "var(--green)" : "var(--red)";
    return `<div class="dist-row">
      <span class="dist-label">${escapeHtml(f.feature)}</span>
      <span class="dist-bar-track"><span class="dist-bar-fill" style="width:${pct}%; background:${color}"></span></span>
      <span class="dist-pct">${f.mean_signed_contribution >= 0 ? "+" : ""}${f.mean_signed_contribution.toFixed(4)}</span>
    </div>`;
  }).join("");
}

async function pollExplanation(jobId) {
  explanationPolling.add(jobId);
  try {
    const job = await api(`/api/explanations/${jobId}`);
    if (job.status === "completed") {
      explanationResults[jobId] = job.result;
      explanationPolling.delete(jobId);
      renderPredictionTab();
    } else if (job.status === "failed") {
      explanationPolling.delete(jobId);
    } else {
      setTimeout(() => { explanationPolling.delete(jobId); pollExplanation(jobId); }, 500);
    }
  } catch (_) {
    explanationPolling.delete(jobId);
  }
}

const CLASS_COLORS = { BENIGN: "var(--green)", DDoS: "var(--red)", DoS: "var(--orange)", PortScan: "var(--amber)" };

/* ==================== Dashboard mini-charts (hand-rolled inline SVG) ==================== */

// B1 — per-window temporal-states timeline. Fetches the row-level detail
// /api/temporal/status deliberately omits; only refetches when the prepared
// session or its window count changed. 404 => nothing prepared yet.
async function loadTemporalStates() {
  const summary = temporalStatus && temporalStatus.result;
  if (!summary) { temporalStatesData = null; temporalStatesKey = null; return; }
  const wantKey = `${summary.output_dir || ""}#${summary.total_windows || 0}`;
  if (wantKey === temporalStatesKey && temporalStatesData) return;
  try {
    temporalStatesData = await api("/api/temporal/states");
    temporalStatesKey = wantKey;
  } catch (e) {
    temporalStatesData = null;   // 404 before a dataset is prepared
    temporalStatesKey = null;
  }
}

function renderTemporalStatesTimeline() {
  const data = temporalStatesData;
  if (!data || !data.rows || !data.rows.length) {
    return `<div id="temporal-states-timeline" class="mini-chart"><div class="dist-empty">No prepared temporal windows yet — run Prepare Temporal Dataset.</div></div>`;
  }
  const rows = data.rows;
  const W = 640, H = 96, padL = 4, padR = 4, padTop = 8, padBot = 18;
  const plotW = W - padL - padR, plotH = H - padTop - padBot;
  const maxFlows = Math.max(1, ...rows.map((r) => r.flow_count || 0));
  const bw = plotW / rows.length;
  const bars = rows.map((r, i) => {
    const h = Math.max(2, (r.flow_count || 0) / maxFlows * plotH);
    const x = padL + i * bw;
    const y = padTop + (plotH - h);
    const fill = CLASS_COLORS[r.dominant_state] || "var(--accent)";
    const tick = r.attack_present
      ? `<rect x="${x.toFixed(1)}" y="${padTop.toFixed(1)}" width="${Math.max(1, bw - 0.5).toFixed(1)}" height="3" fill="var(--red)"/>`
      : "";
    return `<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${Math.max(1, bw - 0.5).toFixed(1)}" height="${h.toFixed(1)}" fill="${fill}"><title>${escapeHtml(String(r.window_start || ("window " + r.window_id)))} · ${escapeHtml(r.dominant_state || "?")} · ${r.flow_count || 0} flows${r.attack_present ? " · attack" : ""}</title></rect>${tick}`;
  }).join("");
  const baseY = padTop + plotH;
  const firstLbl = escapeHtml(String(rows[0].window_start || rows[0].window_id));
  const lastLbl = escapeHtml(String(rows[rows.length - 1].window_start || rows[rows.length - 1].window_id));
  const legend = ["BENIGN", "DoS", "DDoS", "PortScan"]
    .map((c) => `<span><i style="background:${CLASS_COLORS[c]}"></i>${c}</span>`).join("") +
    `<span><i style="background:var(--red)"></i>attack window</span>`;
  return `<div id="temporal-states-timeline" class="mini-chart">
    <svg viewBox="0 0 ${W} ${H}" role="img" aria-label="dominant network state per ${data.window_size_seconds || 10}s window">
      <line x1="${padL}" y1="${baseY}" x2="${padL + plotW}" y2="${baseY}" stroke="currentColor"/>
      ${bars}
      <text x="${padL}" y="${H - 4}" text-anchor="start">${firstLbl}</text>
      <text x="${padL + plotW}" y="${H - 4}" text-anchor="end">${lastLbl}</text>
    </svg>
    <div class="mini-chart-legend">${legend}<span style="margin-left:auto">${escapeHtml(data.session)} · ${data.source === "in_process" ? "live" : "on disk"} · bar height = flow count</span></div>
  </div>`;
}

// B2 — attack vs benign flows per minute + mean confidence, from the
// prediction result already on the page. No backend call.
function renderPredictionFlowsTimeline(pred) {
  const wrap = el("prediction-flows-timeline");
  if (!wrap) return;
  const flows = (pred && pred.flows) || [];
  const parsed = flows.map((f) => {
    const raw = f.timestamp || f.Timestamp || "";
    const ms = Date.parse(String(raw).replace(" ", "T"));
    return Number.isNaN(ms) ? null : { ms, attack: (f.effective_state || f.predicted_state) !== "BENIGN", conf: typeof f.confidence === "number" ? f.confidence : null };
  }).filter(Boolean).sort((a, b) => a.ms - b.ms);
  if (parsed.length < 2) {
    wrap.innerHTML = '<div class="dist-empty">N/A — not enough timestamped flows</div>';
    return;
  }
  const t0 = parsed[0].ms, t1 = parsed[parsed.length - 1].ms;
  const span = Math.max(1, t1 - t0);
  const nbins = Math.min(120, Math.max(1, Math.ceil(span / 60000)));
  const bins = Array.from({ length: nbins }, () => ({ benign: 0, attack: 0, confSum: 0, confN: 0 }));
  for (const p of parsed) {
    const idx = Math.min(nbins - 1, Math.floor((p.ms - t0) / span * nbins));
    if (p.attack) bins[idx].attack++; else bins[idx].benign++;
    if (p.conf != null) { bins[idx].confSum += p.conf; bins[idx].confN++; }
  }
  const W = 640, H = 150, padL = 34, padR = 8, padTop = 10, padBot = 22;
  const plotW = W - padL - padR, plotH = H - padTop - padBot;
  const maxCount = Math.max(1, ...bins.map((b) => Math.max(b.benign, b.attack)));
  const xAt = (i) => padL + (nbins === 1 ? plotW / 2 : i * (plotW / (nbins - 1)));
  const yCount = (v) => padTop + (1 - v / maxCount) * plotH;
  const yConf = (v) => padTop + (1 - v) * plotH;
  const line = (pick, color, dash) => `<polyline points="${bins.map((b, i) => `${xAt(i).toFixed(1)},${pick(b).toFixed(1)}`).join(" ")}" fill="none" stroke="${color}" stroke-width="2"${dash ? ` stroke-dasharray="4 3"` : ""}/>`;
  const benignLine = line((b) => yCount(b.benign), "var(--green)");
  const attackLine = line((b) => yCount(b.attack), "var(--red)");
  const confLine = line((b) => yConf(b.confN ? b.confSum / b.confN : 0), "var(--amber)", true);
  const fmtT = (ms) => new Date(ms).toISOString().slice(11, 19);
  wrap.innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="attack vs benign flows per minute and mean confidence">
    <line x1="${padL}" y1="${padTop + plotH}" x2="${padL + plotW}" y2="${padTop + plotH}" stroke="currentColor"/>
    <line x1="${padL}" y1="${padTop}" x2="${padL}" y2="${padTop + plotH}" stroke="currentColor"/>
    <text x="${padL - 4}" y="${padTop + 4}" text-anchor="end">${maxCount}</text>
    <text x="${padL - 4}" y="${padTop + plotH}" text-anchor="end">0</text>
    ${benignLine}${attackLine}${confLine}
    <text x="${padL}" y="${H - 6}" text-anchor="start">${fmtT(t0)}</text>
    <text x="${padL + plotW}" y="${H - 6}" text-anchor="end">${fmtT(t1)}</text>
  </svg>
  <div class="mini-chart-legend">
    <span><i style="background:var(--green)"></i>benign flows/min</span>
    <span><i style="background:var(--red)"></i>attack flows/min</span>
    <span><i style="background:var(--amber)"></i>mean confidence (0–1)</span>
  </div>`;
}

// B3 — packets/sec sparkline for a live capture, accumulated client-side
// from the poll that already runs. No backend call.
function renderCaptureRateSparkline() {
  const wrap = el("capture-rate-sparkline");
  if (!wrap) return;
  const hist = packetRateHistory;
  if (hist.length < 2) { wrap.innerHTML = '<div class="dist-empty">&mdash;</div>'; return; }
  const rates = [];
  for (let i = 1; i < hist.length; i++) {
    const dt = (hist[i].t - hist[i - 1].t) / 1000;
    const dp = hist[i].packets - hist[i - 1].packets;
    rates.push(dt > 0 && dp >= 0 ? dp / dt : 0);
  }
  const W = 320, H = 48, pad = 4;
  const maxR = Math.max(1, ...rates);
  const stepX = (W - pad * 2) / Math.max(1, rates.length - 1);
  const pts = rates.map((r, i) => `${(pad + i * stepX).toFixed(1)},${(H - pad - r / maxR * (H - pad * 2)).toFixed(1)}`).join(" ");
  const current = rates[rates.length - 1] || 0;
  wrap.innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="packets per second">
    <polyline points="${pts}" fill="none" stroke="var(--green)" stroke-width="2"/>
    <text x="${W - pad}" y="${pad + 9}" text-anchor="end">${current.toFixed(0)} pkt/s</text>
  </svg>`;
}
function renderClassDistribution(pred) {
  const wrap = el("class-distribution");
  if (!pred || !pred.class_counts) {
    wrap.innerHTML = '<div class="dist-empty">N/A — no prediction yet</div>';
    return;
  }
  const total = pred.flows_analyzed || Object.values(pred.class_counts).reduce((a, b) => a + b, 0) || 1;
  wrap.innerHTML = Object.entries(pred.class_counts).map(([cls, count]) => {
    const pct = Math.round((count / total) * 100);
    return `<div class="dist-row">
      <span class="dist-label">${cls}</span>
      <span class="dist-bar-track"><span class="dist-bar-fill" style="width:${pct}%; background:${CLASS_COLORS[cls] || "var(--accent)"}"></span></span>
      <span class="dist-pct">${count} (${pct}%)</span>
    </div>`;
  }).join("");
}

function renderTemporalTab() {
  el("t-window-size").textContent = temporalStatus.result ? temporalStatus.result.window_size_seconds + " seconds" : "10 seconds (default)";
  el("t-sequence-length").textContent = temporalStatus.result ? temporalStatus.result.sequence_length : "5 (default)";
  el("t-status").textContent = temporalStatus.stage;

  const csvPath = currentExtractionOutputCsv();
  el("btn-temporal").disabled = !csvPath || temporalPolling;

  const steps = ["csv", "window", "state", "transition", "sequence", "split"];
  const doneIdx = temporalStatus.stage === "COMPLETED" ? steps.length : (temporalStatus.stage === "PREPARING" ? Math.ceil(steps.length / 2) : 0);
  steps.forEach((s, i) => {
    const stepEl = document.querySelector(`.temporal-step[data-tstep="${s}"]`);
    stepEl.className = "temporal-step" + (temporalStatus.stage === "COMPLETED" ? " tstep-done" : (temporalStatus.stage === "PREPARING" && i < doneIdx ? " tstep-active" : ""));
  });

  const r = temporalStatus.result;
  el("t-windows").textContent = r ? r.total_windows : "N/A";
  el("t-features").textContent = r ? r.state_features : "N/A";
  el("t-transitions").textContent = r ? r.transitions : "N/A";
  el("t-sequences").textContent = r ? r.total_sequences : "N/A";
  el("t-split-windows").textContent = r ? `${r.train_windows} / ${r.validation_windows} / ${r.test_windows}` : "N/A";
  el("t-split-sequences").textContent = r ? `${r.train_sequences} / ${r.validation_sequences} / ${r.test_sequences}` : "N/A";

  renderValidationTab();
  renderLstmTab();
}

let benchmarkData = null;
async function loadBenchmark() {
  if (benchmarkData) return;
  try { benchmarkData = await api("/api/benchmark"); renderBenchmark(); } catch (e) { /* leave */ }
}
function renderBenchmark() {
  const wrap = el("lstm-evaluation");
  if (!wrap) return;
  const b = benchmarkData && benchmarkData.one_step && benchmarkData.one_step.validation;
  if (!b) { wrap.innerHTML = '<div class="dist-empty">Loading benchmark…</div>'; loadBenchmark(); return; }
  const f = (x) => (x == null || typeof x !== "number") ? "N/A" : x.toFixed(4);
  const row = (label, k) => `<tr><td>${label}</td><td>${f(b.lstm[k])}</td><td>${f(b.logistic_regression[k])}</td></tr>`;
  wrap.innerHTML = `<div class="detail-col-subtitle">LSTM vs Logistic Regression &mdash; validation split (frozen Phase&nbsp;3 evaluation)</div>
    <table class="data-table" style="width:100%"><thead><tr><th>Metric</th><th>LSTM</th><th>Logistic Reg.</th></tr></thead><tbody>
    ${row("Macro F1", "macro_f1")}
    ${row("Macro precision", "macro_precision")}
    ${row("Macro recall", "macro_recall")}
    ${row("Weighted F1", "weighted_f1")}
    ${row("Attack F1", "attack_f1")}
    ${row("Attack precision", "attack_precision")}
    ${row("Attack recall", "attack_recall")}
    ${row("Attack false-positive rate", "attack_false_positive_rate")}
    </tbody></table>
    <div class="dist-empty" style="text-align:left">${escapeHtml(benchmarkData.note || "")}</div>`;
}

function renderLstmTab() {
  const status = lstmStatus || {};
  el("lstm-stage").textContent = String(status.stage || "idle").toUpperCase();
  el("lstm-rows").textContent = Number(status.rows_processed || 0).toLocaleString();
  el("lstm-cache").textContent = String(status.cache_state || "unknown").toUpperCase();
  el("lstm-epoch").textContent = status.epoch || 0;
  el("lstm-val-f1").textContent = status.validation_macro_f1 == null ? "N/A" : Number(status.validation_macro_f1).toFixed(4);
  el("btn-lstm-train").disabled = lstmPolling || !["idle", "completed", "error"].includes(status.stage || "idle");
  el("btn-lstm-report").disabled = !lstmReport;
  el("btn-lstm-forecast").disabled = !lstmReport;

  const error = el("lstm-error");
  error.style.display = status.error ? "" : "none";
  error.textContent = status.error || "";

  const metrics = lstmReport && lstmReport.metrics && lstmReport.metrics.holdout;
  if (metrics) {
    const value = (name, key) => metrics[name][key] == null ? "N/A" : Number(metrics[name][key]).toFixed(4);
    el("lstm-evaluation").innerHTML = `<table class="data-table" style="width:100%"><thead><tr><th>Model</th><th>Accuracy</th><th>Macro-F1</th><th>Weighted-F1</th></tr></thead><tbody>
      <tr><td>LSTM</td><td>${value("lstm", "accuracy")}</td><td>${value("lstm", "macro_f1")}</td><td>${value("lstm", "weighted_f1")}</td></tr>
      <tr><td>Persistence</td><td>${value("persistence", "accuracy")}</td><td>${value("persistence", "macro_f1")}</td><td>${value("persistence", "weighted_f1")}</td></tr>
      <tr><td>Logistic Regression</td><td>${value("logistic_regression", "accuracy")}</td><td>${value("logistic_regression", "macro_f1")}</td><td>${value("logistic_regression", "weighted_f1")}</td></tr>
      </tbody></table><div>Evaluation: ${escapeHtml(lstmReport.evaluation_status)}</div>
      <div>Rows / windows / sequences: ${Number(lstmReport.counts.rows).toLocaleString()} / ${Number(lstmReport.counts.windows).toLocaleString()} / ${Number(lstmReport.counts.train_sequences + lstmReport.counts.holdout_sequences).toLocaleString()}</div>`;
  } else {
    renderBenchmark();
  }

  const forecast = lstmForecast;
  el("lstm-current").textContent = forecast ? forecast.current_state : "N/A";
  el("lstm-predicted").textContent = forecast ? forecast.predicted_state : "N/A";
  el("lstm-alert").textContent = forecast ? (forecast.attack_alert_forecast || "NONE") : "N/A";
  el("lstm-confidence").textContent = forecast ? `${(forecast.confidence * 100).toFixed(2)}%` : "N/A";
  el("lstm-probabilities").innerHTML = forecast ? Object.entries(forecast.probabilities).map(([label, probability]) => {
    const percent = Math.round(probability * 100);
    return `<div class="dist-row"><span class="dist-label">${label}</span><span class="dist-bar-track"><span class="dist-bar-fill" style="width:${percent}%; background:${CLASS_COLORS[label]}"></span></span><span class="dist-pct">${percent}%</span></div>`;
  }).join("") : '<div class="dist-empty">N/A</div>';
  const mapping = forecast && forecast.mitre_mapping;
  const supportedCandidates = mapping ? mapping.mitre_candidates.filter((candidate) => candidate.mapping_status === "POSSIBLE") : [];
  const supported = supportedCandidates.length ? supportedCandidates.map((candidate) => `
      <div class="mitre-candidate">
        <div class="mitre-candidate-heading">${escapeHtml(candidate.technique_id)} — ${escapeHtml(candidate.technique_name)}</div>
        <div><span>Tactic</span><strong>${escapeHtml(candidate.tactic)}</strong></div>
        <div><span>Mapping status</span><strong>${escapeHtml(candidate.mapping_status)}</strong></div>
        <div><span>Mapping confidence</span><strong>${(candidate.mapping_confidence * 100).toFixed(0)}%</strong></div>
        <div class="mitre-copy"><span>Evidence</span><ul>${candidate.evidence.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul></div>
        <div class="mitre-copy"><span>Limitations</span><ul>${candidate.limitations.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul></div>
      </div>`).join("") : `<div class="dist-empty">${escapeHtml(mapping ? mapping.reason : "No forecast context available.")}</div>`;
  const evidence = mapping && mapping.evidence_needed ? mapping.evidence_needed.map(item => `<li>${escapeHtml(item)}</li>`).join("") : "<li>Run a forecast to identify evidence gaps.</li>";
  const actions = mapping && mapping.operator_guidance ? mapping.operator_guidance.map(item => `<li>${escapeHtml(item)}</li>`).join("") : "<li>Continue monitoring.</li>";
  el("lstm-mitre-context").innerHTML = `
    <div class="mitre-context-label">Supported ATT&amp;CK Context</div>${supported}
    <div class="mitre-context-label">Evidence Needed</div><div class="mitre-candidate mitre-copy"><ul>${evidence}</ul></div>
    <div class="mitre-context-label">Recommended Actions</div><div class="mitre-candidate mitre-copy"><ul>${actions}</ul></div>`;
}

function renderMultistepTab() {
  const forecast = multistepForecast;
  el("multi-current").textContent = forecast ? forecast.current_state : "N/A";
  if (!forecast) return;
  const earliest = forecast.earliest_predicted_attack_horizon;
  el("multi-warning").textContent = earliest
    ? `EARLY WARNING: attack probability reaches ${(forecast.early_warning_threshold * 100).toFixed(0)}% at H${earliest}.`
    : `NO EARLY WARNING at the frozen ${(forecast.early_warning_threshold * 100).toFixed(0)}% threshold.`;
  el("multi-warning").className = "multistep-warning " + (earliest ? "warning-active" : "");
  const width = 640, height = 170, left = 38, top = 16, plotW = 580, plotH = 120;
  const points = (label) => forecast.horizons.map((item, index) => {
    const x = left + index * (plotW / 5);
    const y = top + (1 - item.probabilities[label]) * plotH;
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(" ");
  const lines = forecast.classes.map((label) => `<polyline points="${points(label)}" fill="none" stroke="${CLASS_COLORS[label]}" stroke-width="2"/>`).join("");
  const labels = forecast.horizons.map((item, index) => `<text x="${left + index * (plotW / 5)}" y="${top + plotH + 18}" text-anchor="middle">H${item.horizon}</text>`).join("");
  el("multi-chart").innerHTML = `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="H1 to H6 class probability trajectories"><line x1="${left}" y1="${top + plotH}" x2="${left + plotW}" y2="${top + plotH}" stroke="currentColor"/>${lines}${labels}</svg>`;
  el("multi-horizons").innerHTML = forecast.horizons.map((item) => {
    const candidates = item.mitre_candidates || [];
    const probabilities = Object.entries(item.probabilities).map(([label, value]) => `<li>${escapeHtml(label)}: ${(value * 100).toFixed(2)}%</li>`).join("");
    const mitre = candidates.length ? candidates.map((candidate) => `<li>${escapeHtml(candidate.technique_id)} — ${escapeHtml(candidate.technique_name)} (${(candidate.mapping_confidence * 100).toFixed(0)}% mapping confidence)</li>`).join("") : "<li>No supported network-only ATT&amp;CK candidate.</li>";
    return `<details class="multistep-horizon"><summary><strong>H${item.horizon}</strong><span>dominant ${escapeHtml(item.dominant_state_forecast || item.predicted_state)} · alert ${escapeHtml(item.attack_alert_forecast || "NONE")}</span><span>${(item.forecast_probability * 100).toFixed(2)}%</span><span>attack ${(item.attack_probability * 100).toFixed(2)}%</span></summary><div><strong>Dominant-state probabilities</strong><ul>${probabilities}</ul><strong>Supported ATT&amp;CK Context</strong><ul>${mitre}</ul></div></details>`;
  }).join("");
}

let worldModelForecast = null;
async function runWorldModelForecast() {
  const btn = el("btn-worldmodel-forecast");
  const k = parseInt(el("wm-k").value, 10) || 6;
  btn.disabled = true;
  el("wm-error").style.display = "none";
  try {
    worldModelForecast = await api("/api/worldmodel/forecast", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ k }),
    });
    renderWorldModelTab();
  } catch (e) {
    el("wm-error").style.display = "";
    el("wm-error").textContent = e.message || "forecast failed";
  } finally {
    btn.disabled = false;
  }
}

function renderWorldModelTab() {
  const f = worldModelForecast;
  el("wm-current").textContent = f ? f.current_state : "N/A";
  el("wm-current-stage").textContent = f ? f.current_mitre_stage : "N/A";
  if (!f) return;
  const steps = f.k_steps || [];
  const alarm = f.earliest_alarm_step;
  el("wm-warning").textContent = alarm
    ? `EARLY WARNING: infiltration probability reaches ${(f.early_warning_threshold * 100).toFixed(0)}% at step ${alarm} (~${alarm * f.window_seconds}s ahead).`
    : `NO EARLY WARNING across ${steps.length} steps · max infiltration ${(f.maximum_infiltration_probability * 100).toFixed(1)}%`;
  el("wm-warning").className = "multistep-warning " + (alarm ? "warning-active" : "");

  // infiltration-probability curve
  const w = 640, h = 160, left = 40, top = 14, plotW = 560, plotH = 110;
  const n = Math.max(1, steps.length - 1);
  const pts = steps.map((s, i) => {
    const x = left + i * (plotW / n);
    const y = top + (1 - s.infiltration_probability) * plotH;
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(" ");
  const xlabels = steps.map((s, i) => `<text x="${left + i * (plotW / n)}" y="${top + plotH + 16}" text-anchor="middle">+${s.offset_seconds}s</text>`).join("");
  el("wm-chart").innerHTML = `<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="infiltration probability over K steps">
    <line x1="${left}" y1="${top + plotH}" x2="${left + plotW}" y2="${top + plotH}" stroke="currentColor"/>
    <polyline points="${pts}" fill="none" stroke="var(--red)" stroke-width="2"/>${xlabels}</svg>`;

  const ps = f.predicted_mitre_stage || {};
  const ladder = (ps.progress_stages || []).map(s => `<span class="${s.active ? "wm-stage-active" : ""}">${escapeHtml(s.name)}</span>`).join(" → ");
  el("wm-stage").innerHTML = `Predicted stage: <strong>${escapeHtml(ps.predicted_stage || "—")}</strong> (${escapeHtml(ps.tactic_id || "n/a")})` +
    (ps.terminal_impact_alert ? ' <span style="color:var(--red)">· TERMINAL IMPACT ALERT</span>' : "") +
    `<div style="margin-top:4px;">${ladder}</div>`;

  const feats = f.top_features || [];
  const fmax = Math.max(...feats.map(x => Math.abs(x.mean_abs_contribution)), 1e-9);
  el("wm-features").innerHTML = feats.length ? feats.map(x => {
    const pct = Math.round((Math.abs(x.mean_abs_contribution) / fmax) * 100);
    const color = x.mean_signed_contribution >= 0 ? "var(--red)" : "var(--green)";
    return `<div class="dist-row"><span class="dist-label">${escapeHtml(x.feature)}</span>
      <span class="dist-bar-track"><span class="dist-bar-fill" style="width:${pct}%; background:${color}"></span></span>
      <span class="dist-pct">${x.mean_signed_contribution >= 0 ? "+" : ""}${x.mean_signed_contribution.toFixed(4)}</span></div>`;
  }).join("") : '<div class="dist-empty">none</div>';

  const wins = f.top_windows || [];
  el("wm-windows").innerHTML = wins.length ? wins.map(x =>
    `<div class="dist-row"><span class="dist-label">window ${x.window_id} (t${x.position})</span>
     <span class="dist-bar-track"><span class="dist-bar-fill" style="width:${Math.round(x.attention * 100)}%"></span></span>
     <span class="dist-pct">${(x.attention * 100).toFixed(1)}%</span></div>`).join("") : '<div class="dist-empty">none</div>';

  el("wm-steps").innerHTML = steps.map(s => {
    const probs = Object.entries(s.probabilities).map(([k2, v]) => `<li>${escapeHtml(k2)}: ${(v * 100).toFixed(1)}%</li>`).join("");
    return `<details class="multistep-horizon"><summary><strong>step ${s.step}</strong>
      <span>+${s.offset_seconds}s</span><span>${escapeHtml(s.predicted_state)}</span>
      <span>infil ${(s.infiltration_probability * 100).toFixed(1)}%</span><span>${escapeHtml(s.risk_level)}</span></summary>
      <div><strong>Probabilities</strong><ul>${probs}</ul>MITRE stage: ${escapeHtml(s.mitre_stage)}</div></details>`;
  }).join("");
}

const VALIDATION_STAGE_LABEL = {
  NOT_VALIDATED: "NOT VALIDATED", VALIDATING: "VALIDATING…",
  VALIDATED: "VALIDATED", VALIDATED_WITH_WARNINGS: "VALIDATED WITH WARNINGS",
  VALIDATION_FAILED: "VALIDATION FAILED", ERROR: "ERROR",
};
const VALIDATION_STAGE_CLASS = {
  NOT_VALIDATED: "v-pending", VALIDATING: "v-pending",
  VALIDATED: "v-pass", VALIDATED_WITH_WARNINGS: "v-warning",
  VALIDATION_FAILED: "v-fail", ERROR: "v-fail",
};

function renderValidationTab() {
  // /api/temporal/validate takes no body — it derives source_csv/temporal_dir
  // from the server's own last-completed temporal_state, so this only needs
  // that a temporal dataset was prepared (in this server process), not that
  // the current UI mode/session still has a matching extraction CSV.
  const canValidate = temporalStatus.stage === "COMPLETED" && !validationPolling;
  el("btn-validate").disabled = !canValidate;

  el("v-overall-status").textContent = VALIDATION_STAGE_LABEL[validationStatus.stage] || validationStatus.stage;
  el("v-overall-status").className = "validation-overall-status " + (VALIDATION_STAGE_CLASS[validationStatus.stage] || "v-pending");

  const report = validationStatus.report;
  el("btn-download-report").disabled = !report;

  if (!report) {
    el("v-rows-checked").textContent = "N/A";
    el("v-windows").textContent = "N/A";
    el("v-sequences").textContent = "N/A";
    el("v-check-counts").textContent = "N/A";
    el("leakage-banner").style.display = "none";
    el("validation-grid-tbody").innerHTML = '<tr class="empty-row"><td colspan="3">Not validated yet.</td></tr>';
    el("v-cov-start").textContent = "N/A"; el("v-cov-end").textContent = "N/A"; el("v-cov-duration").textContent = "N/A";
    el("v-distribution").innerHTML = '<div class="dist-empty">N/A</div>';
    el("v-split-box").textContent = "N/A";
    el("v-feature-table").innerHTML = "";
    el("validation-detail-body").textContent =
      validationStatus.stage === "ERROR" ? ("ERROR: " + validationStatus.error) : "Click a check above to see its detail.";
    return;
  }

  const d = report.details;
  el("v-rows-checked").textContent = report.rows_checked;
  el("v-windows").textContent = d.windows && d.windows.details.total_windows !== undefined ? d.windows.details.total_windows : "N/A";
  el("v-sequences").textContent = d.sequences && d.sequences.details.total_sequences !== undefined ? d.sequences.details.total_sequences : "N/A";
  const counts = { PASS: 0, WARNING: 0, FAIL: 0, NOT_AVAILABLE: 0 };
  Object.values(report.checks).forEach((s) => { counts[s] = (counts[s] || 0) + 1; });
  el("v-check-counts").textContent = `${counts.PASS} PASS | ${counts.WARNING} WARNING | ${counts.FAIL} FAIL`;

  // Leakage banner
  const leak = d.leakage;
  const bannerEl = el("leakage-banner");
  if (leak && leak.status !== "NOT_AVAILABLE") {
    const ld = leak.details;
    const totalOverlap = Object.values(ld.exact_duplicate_sequences).reduce((a, b) => a + b, 0)
      + Object.values(ld.overlapping_window_sequences).reduce((a, b) => a + b, 0);
    if (leak.status === "FAIL") {
      bannerEl.className = "leakage-banner";
      bannerEl.innerHTML = `<div class="leakage-banner-title">&#9888; DATA LEAKAGE DETECTED</div>
        ${totalOverlap} overlapping/duplicate sequence(s) found across TRAIN/VALIDATION/TEST splits
        ${ld.label_leakage === "DETECTED" ? " and label-leakage fields were found in the feature set" : ""}.
        Model training should not proceed until this is resolved.`;
    } else {
      bannerEl.className = "leakage-banner v-ok";
      bannerEl.innerHTML = `&#10003; NO CROSS-SPLIT LEAKAGE DETECTED`;
    }
    bannerEl.style.display = "";
  } else {
    bannerEl.style.display = "none";
  }

  // Check grid
  el("validation-grid-tbody").innerHTML = CHECK_ORDER.map(([key, label]) => {
    const c = d[key];
    const detail = _checkSummaryText(key, c);
    return `<tr data-check-key="${key}" class="${selectedValidationCheck === key ? "v-row-selected" : ""}">
      <td>${label}</td><td class="v-status-${c.status}">${statusIcon(c.status)} ${c.status}</td><td>${escapeHtml(detail)}</td>
    </tr>`;
  }).join("");
  document.querySelectorAll("#validation-grid-tbody tr[data-check-key]").forEach((row) => {
    row.addEventListener("click", () => {
      selectedValidationCheck = row.dataset.checkKey;
      renderValidationTab();
    });
  });

  // Detail panel
  if (selectedValidationCheck && d[selectedValidationCheck]) {
    const c = d[selectedValidationCheck];
    el("validation-detail-title").textContent = CHECK_ORDER.find((x) => x[0] === selectedValidationCheck)[1].toUpperCase() + " VALIDATION";
    el("validation-detail-body").textContent = JSON.stringify(c.details, null, 2);
  }

  // Temporal coverage (from timestamps check)
  const ts = d.timestamps;
  if (ts && ts.details) {
    el("v-cov-start").textContent = na(ts.details.first_timestamp);
    el("v-cov-end").textContent = na(ts.details.last_timestamp);
    el("v-cov-duration").textContent = ts.details.duration_seconds != null ? fmtDuration(ts.details.duration_seconds) : "N/A";
  }

  // Current-state distribution (from current_state check — real label counts, independent of prediction JSON)
  const cs = d.current_state;
  if (cs && cs.details && cs.details.distribution) {
    el("v-distribution").innerHTML = Object.entries(cs.details.distribution).map(([cls, info]) => `
      <div class="dist-row">
        <span class="dist-label">${cls}</span>
        <span class="dist-bar-track"><span class="dist-bar-fill" style="width:${info.percent}%; background:${CLASS_COLORS[cls] || "var(--accent)"}"></span></span>
        <span class="dist-pct">${info.count} (${info.percent}%)</span>
      </div>`).join("");
  }

  // Chronological split box
  const split = d.chronological_split;
  if (split && split.status !== "NOT_AVAILABLE") {
    const sd = split.details;
    el("v-split-box").innerHTML = ["train", "validation", "test"].map((k) => `
      <div class="split-block-title">${k.toUpperCase()}</div>
      Start: ${na(sd[k] && sd[k].start)}<br/>End: ${na(sd[k] && sd[k].end)}<br/>Sequences: ${na(sd[k] && sd[k].sequences)}
    `).join("") + `<div style="margin-top:6px;">Chronological Order: <b class="${sd.order_valid ? "v-status-PASS" : "v-status-FAIL"}">${sd.order_valid ? "PASS" : "FAIL"}</b></div>`;
  } else {
    el("v-split-box").textContent = "N/A — split artifacts not available.";
  }

  // 28-feature table
  const feat = d.features;
  if (feat && feat.details && feat.details.feature_table) {
    el("v-feature-table").innerHTML = `<table class="data-table" style="width:100%;">
      <thead><tr><th>Feature</th><th>Type</th><th>NaN</th><th>Inf</th><th>Min</th><th>Max</th><th>Status</th></tr></thead>
      <tbody>${feat.details.feature_table.map((f) => `
        <tr><td>${escapeHtml(f.feature)}</td><td>${escapeHtml(f.type)}</td><td>${na(f.nan)}</td><td>${na(f.infinite)}</td>
        <td>${f.min != null ? f.min.toFixed(2) : "N/A"}</td><td>${f.max != null ? f.max.toFixed(2) : "N/A"}</td>
        <td class="v-status-${f.status}">${statusIcon(f.status)}</td></tr>`).join("")}</tbody>
    </table>`;
  }
}

function _checkSummaryText(key, c) {
  const d = c.details || {};
  switch (key) {
    case "current_state": return `${report_rows(d)} rows, ${d.unknown_label_count || 0} unknown`;
    case "timestamps": return d.order_status === "FAIL" ? `${d.out_of_order_rows} out of order` : "Ordered";
    case "windows": return `${na(d.total_windows)} windows`;
    case "features": return `${na(d.features_checked)} checked, ${na(d.nan_values)} NaN`;
    case "transitions": return `${na(d.total_transitions)} transitions`;
    case "sequences": return `${na(d.total_sequences)} sequences`;
    case "chronological_split": return d.order_valid === undefined ? "N/A" : (d.order_valid ? "Valid" : "Invalid");
    case "leakage": return c.status === "PASS" ? "None" : "Overlap detected";
    case "missing_data": return `${na(d.rows_checked)} rows`;
    case "duplicates": return `${na(d.exact_duplicate_rows)} duplicate rows`;
    default: return "";
  }
}
function report_rows(d) { return d.rows_checked != null ? d.rows_checked : "N/A"; }

/* ==================== Footer / status bar ==================== */
function renderStatusBar() {
  const cap = mode === "live" ? (liveSnap && liveSnap.capture) : null;
  const pred = mode === "live" ? (liveSnap && liveSnap.prediction) : (uploadStatus && uploadStatus.prediction);
  const packetCount = mode === "live" ? (cap && cap.packet_count != null ? cap.packet_count : 0) : (uploadStatus && uploadStatus.packet_count != null ? uploadStatus.packet_count : 0);
  el("sb-packets").textContent = `Packets: ${packetCount}`;
  el("sb-flows").textContent = `Flows: ${pred ? pred.flows_analyzed : "—"}`;
  const sbv = predVerdict(pred);
  el("sb-state").textContent = `State: ${
    !pred ? "—" :
    sbv.tier === "attack" ? "ATTACK/" + sbv.cls :
    sbv.tier === "suspicious" ? "SUSPICIOUS/" + sbv.cls + "?" :
    "BENIGN"
  }`;
  el("sb-temporal").textContent = `Temporal: ${
    temporalStatus.stage === "COMPLETED" ? `READY (${temporalStatus.result.total_windows}w/${temporalStatus.result.total_sequences}s)` :
    temporalStatus.stage === "PREPARING" ? "PREPARING" :
    temporalStatus.stage === "ERROR" ? "ERROR" : "—"
  }`;

  el("hdr-session-val").textContent =
    mode === "live" ? (cap ? cap.session_id : "—") : (uploadSessionId || "—");

  el("btn-export-prediction").disabled = !(mode === "upload" && uploadStatus && uploadStatus.stage === "PREDICTION_COMPLETED");
  el("btn-export-pcap").disabled = true;   // no download endpoint exists for raw PCAP in either mode
  el("btn-export-features").disabled = true; // no download endpoint exists for the raw feature CSV in either mode
}

/* ==================== Top-level render ==================== */
function renderAll() {
  renderPipelineStrip();
  renderCaptureTab();
  renderExtractionTab();
  renderPredictionTab();
  renderTemporalTab();
  renderMultistepTab();
  renderWorldModelTab();
  renderResponseTab();
  renderActiveTable();
  renderStatusBar();
}

/* ==================== Resizer ==================== */
function initResizer() {
  const resizer = el("resizer");
  const tablePanel = document.querySelector(".table-panel");
  let dragging = false;
  resizer.addEventListener("mousedown", () => { dragging = true; document.body.style.cursor = "row-resize"; });
  window.addEventListener("mouseup", () => { dragging = false; document.body.style.cursor = ""; });
  window.addEventListener("mousemove", (e) => {
    if (!dragging) return;
    const mainRect = document.querySelector(".workspace").getBoundingClientRect();
    const newHeight = e.clientY - mainRect.top;
    if (newHeight > 100 && newHeight < mainRect.height - 150) {
      tablePanel.style.flex = `0 0 ${newHeight}px`;
    }
  });
}

/* ==================== Wiring ==================== */
el("mode-live").addEventListener("click", () => switchMode("live"));
el("mode-upload").addEventListener("click", () => switchMode("upload"));
el("upload-file-input").addEventListener("change", (e) => {
  selectedUploadFile = e.target.files[0] || null;
  el("btn-analyze").disabled = !selectedUploadFile;
  el("upload-file-info").textContent = selectedUploadFile
    ? `${selectedUploadFile.name} — ${fmtBytes(selectedUploadFile.size)}` : "No file selected";
});
el("btn-analyze").addEventListener("click", analyzeFile);
el("btn-download-csv").addEventListener("click", downloadProcessedCsv);
el("btn-export-prediction").addEventListener("click", downloadProcessedCsv);
el("btn-start").addEventListener("click", startCapture);
el("iface-all").addEventListener("change", () => { el("iface-select").disabled = el("iface-all").checked; });
el("btn-stop").addEventListener("click", stopCapture);
el("btn-packet-features").addEventListener("click", loadPacketFeatures);
el("btn-classifier-metrics").addEventListener("click", loadClassifierMetrics);
el("btn-extract").addEventListener("click", runExtract);
el("btn-predict").addEventListener("click", runPredict);
el("btn-reset").addEventListener("click", resetPipeline);
el("btn-temporal").addEventListener("click", prepareTemporalDataset);
el("btn-validate").addEventListener("click", runValidation);
el("btn-download-report").addEventListener("click", downloadValidationReport);
el("btn-lstm-train").addEventListener("click", startLstmTraining);
el("btn-lstm-forecast").addEventListener("click", runLstmForecast);
el("btn-lstm-report").addEventListener("click", downloadLstmReport);
el("btn-multistep-forecast").addEventListener("click", runMultistepForecast);
el("btn-worldmodel-forecast").addEventListener("click", runWorldModelForecast);
el("btn-response-scan").addEventListener("click", scanFirewall);
el("btn-response-plan").addEventListener("click", createResponsePlan);
el("btn-response-apply").addEventListener("click", applyResponsePlan);
el("btn-response-verify").addEventListener("click", verifyResponseAction);
el("btn-response-rollback").addEventListener("click", rollbackResponseAction);
el("response-apply-ack").addEventListener("change", renderResponseTab);
el("response-rollback-ack").addEventListener("change", renderResponseTab);
el("response-history-filter").addEventListener("change", renderResponseTab);
el("response-event-filter").addEventListener("change", renderResponseTab);
el("xdr-operator-ack").addEventListener("change", renderXdrCampaign);
el("btn-xdr-apply").addEventListener("click", applyXdrDryRun);
el("btn-xdr-rollback").addEventListener("click", rollbackXdrDryRun);
el("btn-packets-prev").addEventListener("click", () => goToPacketsPage(-1));
el("btn-packets-next").addEventListener("click", () => goToPacketsPage(1));

document.querySelectorAll(".table-tab").forEach((b) => b.addEventListener("click", () => switchTableTab(b.dataset.tableTab)));
document.querySelectorAll(".detail-tab").forEach((b) => b.addEventListener("click", () => switchDetailTab(b.dataset.detailTab)));
document.querySelectorAll('#temporal-subtabs .subtab').forEach((b) => b.addEventListener("click", () => switchTemporalSubtab(b.dataset.temporalSubtab)));
document.querySelectorAll(".inner-tab").forEach((b) => b.addEventListener("click", () => {
  activeInnerTab = b.dataset.innerTab;
  document.querySelectorAll(".inner-tab").forEach((t) => t.classList.toggle("inner-tab-active", t.dataset.innerTab === activeInnerTab));
  document.querySelectorAll(".inner-pane").forEach((p) => { p.style.display = p.dataset.innerPane === activeInnerTab ? "" : "none"; });
}));
el("btn-filter-apply").addEventListener("click", renderActiveTable);
el("table-search").addEventListener("keydown", (e) => { if (e.key === "Enter") renderActiveTable(); });
el("table-filter").addEventListener("keydown", (e) => { if (e.key === "Enter") renderActiveTable(); });
el("btn-filter-clear").addEventListener("click", () => { el("table-filter").value = ""; el("table-search").value = ""; renderActiveTable(); });
el("predictions-state-filter").addEventListener("change", renderActiveTable);

el("feature-space-toggle").addEventListener("click", () => {
  const body = el("feature-space-body");
  const open = body.style.display !== "none";
  body.style.display = open ? "none" : "";
  el("feature-space-toggle").innerHTML = (open ? "&#9656;" : "&#9662;") + " FEATURE SPACE (77 model features)";
});

el("btn-help").addEventListener("click", () => { el("help-modal").style.display = "flex"; });
el("btn-help-close").addEventListener("click", () => { el("help-modal").style.display = "none"; });
el("help-modal").addEventListener("click", (e) => { if (e.target.id === "help-modal") el("help-modal").style.display = "none"; });

loadInterfaces();
loadResponseBootstrap();
initResizer();
startPolling();
renderAll();
