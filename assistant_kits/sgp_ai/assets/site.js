const API_BASE = "https://sea-ai-api.azurewebsites.net/pages/sgp-ai";

const form = document.getElementById("query-form");
const queryEl = document.getElementById("query");
const submitButton = document.getElementById("submit");
const stopButton = document.getElementById("stop");
const statusDot = document.getElementById("status-dot");
const statusLabel = document.getElementById("status-label");
const answerEl = document.getElementById("answer");
const answerMeta = document.getElementById("answer-meta");
const sourcesEl = document.getElementById("sources");
const sourceCountEl = document.getElementById("source-count");
const exampleButtons = Array.from(document.querySelectorAll(".examples button"));

let activeController = null;
let backendReady = false;

function setStatus(kind, text) {
  statusDot.className = `dot ${kind}`;
  statusLabel.textContent = text;
  backendReady = kind === "good";
  submitButton.disabled = !backendReady;
}

function setRunning(isRunning) {
  submitButton.disabled = isRunning || !backendReady;
  stopButton.hidden = !isRunning;
}

function readError(response) {
  return response.text().then((text) => {
    try {
      const payload = JSON.parse(text);
      return payload.detail || payload.error || response.statusText;
    } catch (_) {
      return text || response.statusText;
    }
  });
}

function extractText(content) {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) return content.map(extractText).join("");
  if (content && typeof content === "object") {
    if (typeof content.text === "string") return content.text;
    if (content.text && typeof content.text.value === "string") return content.text.value;
    if (content.content) return extractText(content.content);
  }
  return "";
}

function uniqueDocuments(documents) {
  const seen = new Set();
  const clean = [];
  for (const item of Array.isArray(documents) ? documents : []) {
    if (!item || typeof item !== "object") continue;
    const title = String(item.title || item.canonical_title || "").trim();
    const url = String(item.url || "").trim();
    const summary = String(item.summary || "").trim();
    const language = String(item.language || "").trim();
    const year = Number.isInteger(item.year) && item.year > 0 ? item.year : null;
    const key = item.document_id || url || `${title}|${year || ""}`;
    if (!key || seen.has(key)) continue;
    seen.add(key);
    clean.push({ title: title || url || "Untitled document", url, summary, language, year });
  }
  return clean;
}

function renderSources(documents) {
  const clean = uniqueDocuments(documents);
  sourceCountEl.textContent = `${clean.length} document${clean.length === 1 ? "" : "s"}`;
  sourcesEl.replaceChildren();
  if (!clean.length) {
    const empty = document.createElement("p");
    empty.className = "empty";
    empty.textContent = "No references returned yet.";
    sourcesEl.appendChild(empty);
    return;
  }
  for (const doc of clean) {
    const card = document.createElement("article");
    card.className = "source-card";
    const title = document.createElement(doc.url ? "a" : "strong");
    title.textContent = doc.title;
    if (doc.url) {
      title.href = doc.url;
      title.target = "_blank";
      title.rel = "noreferrer noopener";
    }
    card.appendChild(title);
    const metaParts = [];
    if (doc.year) metaParts.push(String(doc.year));
    if (doc.language) metaParts.push(doc.language);
    if (metaParts.length) {
      const meta = document.createElement("div");
      meta.className = "meta";
      meta.textContent = metaParts.join(" · ");
      card.appendChild(meta);
    }
    if (doc.summary) {
      const summary = document.createElement("p");
      summary.textContent = doc.summary;
      card.appendChild(summary);
    }
    sourcesEl.appendChild(card);
  }
}

async function checkStatus() {
  try {
    const response = await fetch(`${API_BASE}/status`, { headers: { "Accept": "application/json" } });
    if (!response.ok) throw new Error(await readError(response));
    const payload = await response.json();
    if (payload.corpus_ready) {
      setStatus("good", `Ready · ${payload.document_count.toLocaleString()} documents`);
    } else {
      setStatus("bad", "Corpus unavailable");
    }
  } catch (error) {
    setStatus("bad", "Backend unavailable");
    answerEl.innerHTML = "";
    const message = document.createElement("p");
    message.className = "error";
    message.textContent = String(error.message || error);
    answerEl.appendChild(message);
  }
}

async function streamAnswer(query, signal) {
  const response = await fetch(`${API_BASE}/model?ui_locale=en`, {
    method: "POST",
    headers: { "Content-Type": "application/json", "Accept": "application/x-ndjson" },
    body: JSON.stringify([{ role: "human", content: query }]),
    signal,
  });
  if (!response.ok) throw new Error(await readError(response));
  if (!response.body) throw new Error("No response stream returned by browser.");

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let answer = "";
  let chunkCount = 0;
  const started = performance.now();

  function handleLine(line) {
    if (!line.trim()) return;
    const payload = JSON.parse(line);
    chunkCount += 1;
    const text = extractText(payload.content);
    if (text) {
      answer += text;
      answerEl.textContent = answer;
    }
    if (Array.isArray(payload.documents)) {
      renderSources(payload.documents);
    }
    answerMeta.textContent = `${Math.round((performance.now() - started) / 100) / 10}s · ${chunkCount} chunks`;
  }

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";
    for (const line of lines) handleLine(line);
  }
  buffer += decoder.decode();
  if (buffer.trim()) handleLine(buffer);
}

async function runQuery(query) {
  const cleanQuery = String(query || "").trim();
  if (!cleanQuery || !backendReady) return;
  if (activeController) activeController.abort();
  activeController = new AbortController();
  answerMeta.textContent = "Streaming...";
  answerEl.textContent = "";
  sourcesEl.innerHTML = '<p class="empty">Waiting for references...</p>';
  sourceCountEl.textContent = "0 documents";
  setRunning(true);
  try {
    await streamAnswer(cleanQuery, activeController.signal);
  } catch (error) {
    if (error.name !== "AbortError") {
      answerEl.innerHTML = "";
      const message = document.createElement("p");
      message.className = "error";
      message.textContent = String(error.message || error);
      answerEl.appendChild(message);
      answerMeta.textContent = "Error";
    }
  } finally {
    setRunning(false);
    activeController = null;
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  runQuery(queryEl.value);
});

stopButton.addEventListener("click", () => {
  if (activeController) activeController.abort();
});

for (const button of exampleButtons) {
  button.addEventListener("click", () => {
    queryEl.value = button.textContent;
    runQuery(button.textContent);
  });
}

checkStatus();
