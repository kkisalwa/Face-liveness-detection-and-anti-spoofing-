/**
 * FaceShield Frontend — Detection Controller
 * Sends browser webcam frames → backend → shows annotated results
 */

const API = "http://localhost:8000";
const WS_URL = "ws://localhost:8000/ws/detect";

let ws = null;
let localStream = null;
let isDetecting = false;
let sendTimer = null;
let vidEl = null;
let canvas = null;
let ctx = null;

// Session counters
let totalDet = 0;
let realCnt = 0;
let spoofCnt = 0;

// ── DOM refs ──────────────────────────────────────────────────────
const $id = id => document.getElementById(id);

const liveFrame = $id("live-frame");
const placeholder = $id("vid-placeholder");
const recBadge = $id("rec-badge");
const liveBadge = $id("live-badge");
const liveLabel = $id("live-label");
const liveDot = $id("live-dot");
const btnStart = $id("btn-start");
const btnStop = $id("btn-stop");
const btnSnap = $id("btn-snap");
const resultCard = $id("result-card");
const resultTitle = $id("result-title");
const resultDesc = $id("result-desc");
const resultPct = $id("result-pct");
const logList = $id("log-list");
const statusBadge = $id("connection-status");
const statusLabel = $id("status-label");

// ── Init ─────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
    checkServer();
    setInterval(checkServer, 8000);
    canvas = document.createElement("canvas");
    ctx = canvas.getContext("2d");
});

// ── Server ping ───────────────────────────────────────────────────
async function checkServer() {
    try {
        const r = await fetch(`${API}/api/status`, { signal: AbortSignal.timeout(3000) });
        if (r.ok) setStatus("online", "Server Online");
        else setStatus("offline", "Server Error");
    } catch {
        setStatus("offline", "Server Offline");
    }
}

function setStatus(cls, text) {
    statusBadge.className = `status-badge ${cls}`;
    statusLabel.textContent = text;
}

// ── Start Detection ───────────────────────────────────────────────
async function startDetection() {
    if (isDetecting) return;

    // Request webcam
    try {
        localStream = await navigator.mediaDevices.getUserMedia({
            video: { width: { ideal: 640 }, height: { ideal: 480 } },
            audio: false,
        });
    } catch (err) {
        alert("Camera access denied or unavailable.\n\nPlease allow camera access and try again.");
        return;
    }

    // Setup hidden video element for frame capture
    vidEl = document.createElement("video");
    vidEl.srcObject = localStream;
    vidEl.playsInline = true;
    vidEl.muted = true;
    vidEl.play();

    vidEl.addEventListener("loadedmetadata", () => {
        canvas.width = vidEl.videoWidth || 640;
        canvas.height = vidEl.videoHeight || 480;
        isDetecting = true;
        setUILive(true);
        startSendLoop();
    });
}

// ── Frame send loop ───────────────────────────────────────────────
function startSendLoop() {
    async function loop() {
        if (!isDetecting) return;
        if (vidEl && vidEl.readyState >= 2) {
            ctx.drawImage(vidEl, 0, 0, canvas.width, canvas.height);
            const dataUrl = canvas.toDataURL("image/jpeg", 0.8);
            await sendFrame(dataUrl);
        }
        sendTimer = setTimeout(loop, 80); // ~12 fps
    }
    loop();
}

async function sendFrame(dataUrl) {
    try {
        const res = await fetch(`${API}/api/analyze`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ image: dataUrl }),
            signal: AbortSignal.timeout(3000),
        });
        if (!res.ok) return;
        const data = await res.json();
        handleResult(data);
    } catch { /* silently skip */ }
}

// ── Handle result ─────────────────────────────────────────────────
function handleResult(data) {
    // Show annotated frame
    if (data.frame) {
        liveFrame.src = data.frame;
        liveFrame.classList.remove("hidden");
        placeholder.style.display = "none";
    }

    // Update server stats if provided
    if (data.stats) updateStatsFromServer(data.stats);

    const results = data.results || [];
    if (results.length > 0) {
        const r = results[0];
        updateResultCard(r);
        updateMeters(r);
        addLogEntry(r);

        // Local counters
        totalDet++;
        if (r.is_real) realCnt++; else spoofCnt++;
        refreshStats();
    } else {
        clearResultCard();
    }
}

// ── Result card ───────────────────────────────────────────────────
function updateResultCard(r) {
    const real = r.is_real;
    resultCard.className = `result-card ${real ? "real-mode" : "spoof-mode"}`;
    resultTitle.textContent = real ? "Real Human Detected" : "Spoof / Fake Detected";
    resultDesc.textContent = real
        ? "Liveness confirmed — genuine face"
        : "Spoofing attempt detected — photo or screen";
    resultPct.textContent = `${r.confidence}%`;

    // Show the right SVG icon
    $id("icon-search").classList.add("hidden");
    $id("icon-check").classList.toggle("hidden", !real);
    $id("icon-warn").classList.toggle("hidden", real);

    // Color the icon box
    const box = $id("result-icon-box");
    box.style.color = real ? "var(--green)" : "var(--red)";
}

function clearResultCard() {
    resultCard.className = "result-card";
    const s = $id("icon-search");
    const ck = $id("icon-check");
    const wn = $id("icon-warn");
    if (s) s.classList.remove("hidden");
    if (ck) ck.classList.add("hidden");
    if (wn) wn.classList.add("hidden");
    const box = $id("result-icon-box");
    if (box) box.style.color = "var(--txt-sec)";
    resultTitle.textContent = "No Face Detected";
    resultDesc.textContent = "Point camera at a face to begin analysis";
    resultPct.textContent = "--";
}

// ── Confidence meters ─────────────────────────────────────────────
function updateMeters(r) {
    const realPct = r.real_confidence ?? r.confidence;
    const spoofPct = r.spoof_confidence ?? (100 - r.confidence);

    $id("real-bar").style.width = clamp(realPct) + "%";
    $id("spoof-bar").style.width = clamp(spoofPct) + "%";
    $id("real-pct").textContent = realPct.toFixed(1) + "%";
    $id("spoof-pct").textContent = spoofPct.toFixed(1) + "%";

    const sc = r.scores || {};
    setFeat("f-texture", "v-texture", sc.texture_entropy, 7.5);
    setFeat("f-sharp", "v-sharp", sc.sharpness, 300);
    setFeat("f-color", "v-color", sc.color_variance, 35);
    setFeat("f-freq", "v-freq", sc.frequency_energy, 12);
}

function setFeat(barId, valId, val, max) {
    if (val === undefined || val === null) return;
    $id(barId).style.width = clamp(val / max * 100) + "%";
    $id(valId).textContent = Number(val).toFixed(1);
}

function clamp(v) { return Math.min(100, Math.max(0, v)); }

// ── Stats ─────────────────────────────────────────────────────────
function refreshStats() {
    $id("s-total").textContent = totalDet;
    $id("s-real").textContent = realCnt;
    $id("s-spoof").textContent = spoofCnt;
    const rate = totalDet > 0 ? Math.round(realCnt / totalDet * 100) : 0;
    $id("s-rate").textContent = totalDet > 0 ? rate + "%" : "—";
}

function updateStatsFromServer(s) {
    $id("s-total").textContent = s.total_detections ?? 0;
    $id("s-real").textContent = s.real_count ?? 0;
    $id("s-spoof").textContent = s.spoof_count ?? 0;
    const tot = (s.real_count ?? 0) + (s.spoof_count ?? 0);
    $id("s-rate").textContent = tot > 0 ? Math.round(s.real_count / tot * 100) + "%" : "—";
}

// ── Detection log ─────────────────────────────────────────────────
function addLogEntry(r) {
    const empty = logList.querySelector(".log-empty");
    if (empty) empty.remove();

    const entry = document.createElement("div");
    entry.className = "log-entry";
    const t = new Date().toLocaleTimeString("en-US", { hour12: false });
    entry.innerHTML = `
    <span class="log-tag ${r.is_real ? "tag-real" : "tag-spoof"}">${r.label}</span>
    <span class="log-desc">${r.is_real ? "Genuine face" : "Spoof attempt"}</span>
    <span class="log-conf">${r.confidence}%</span>
    <span class="log-time">${t}</span>
  `;
    logList.insertBefore(entry, logList.firstChild);
    // Trim to 25 entries
    while (logList.children.length > 25) logList.removeChild(logList.lastChild);
}

function clearLog() {
    logList.innerHTML = '<div class="log-empty">Events will appear here once detection starts...</div>';
}

// ── Stop Detection ────────────────────────────────────────────────
function stopDetection() {
    isDetecting = false;
    if (sendTimer) { clearTimeout(sendTimer); sendTimer = null; }
    if (localStream) { localStream.getTracks().forEach(t => t.stop()); localStream = null; }
    if (ws) { ws.close(); ws = null; }
    fetch(`${API}/api/camera/stop`, { method: "POST" }).catch(() => { });
    setUILive(false);
    clearResultCard();
    liveFrame.classList.add("hidden");
    if (placeholder) placeholder.style.display = "";
}

// ── UI state ──────────────────────────────────────────────────────
function setUILive(active) {
    if (active) {
        btnStart.classList.add("hidden");
        btnStop.classList.remove("hidden");
        btnSnap.disabled = false;
        recBadge.classList.remove("hidden");
        liveBadge.classList.add("active");
        liveLabel.textContent = "LIVE";
    } else {
        btnStart.classList.remove("hidden");
        btnStop.classList.add("hidden");
        btnSnap.disabled = true;
        recBadge.classList.add("hidden");
        liveBadge.classList.remove("active");
        liveLabel.textContent = "STANDBY";
    }
}

// ── Snapshot ──────────────────────────────────────────────────────
function takeSnapshot() {
    if (liveFrame.classList.contains("hidden") || !liveFrame.src) return;
    $id("snap-img").src = liveFrame.src;
    $id("snap-dl").href = liveFrame.src;
    $id("snap-modal").classList.remove("hidden");
}

function closeSnapshot() {
    $id("snap-modal").classList.add("hidden");
}

function closeModal(e) {
    if (e.target === $id("snap-modal")) closeSnapshot();
}

// ── Keyboard shortcuts ────────────────────────────────────────────
document.addEventListener("keydown", e => {
    if (e.key === "Escape") closeSnapshot();
    if (e.key === " " && !["INPUT", "BUTTON", "TEXTAREA"].includes(e.target.tagName)) {
        e.preventDefault();
        isDetecting ? stopDetection() : startDetection();
    }
    if ((e.key === "s" || e.key === "S") && !e.target.matches("input,textarea")) takeSnapshot();
});
