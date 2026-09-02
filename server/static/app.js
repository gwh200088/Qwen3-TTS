/* Qwen3-TTS Voice Clone 前端逻辑（原生 JS，无外部依赖） */
"use strict";

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

const state = {
  voices: [],
  selectedVoiceId: null,
  resultUrl: null,
  ready: false,
  busy: false,
};

/* ---------------------------------------------------------------- */
/* 基础请求封装                                                       */
/* ---------------------------------------------------------------- */
function getApiKey() { return localStorage.getItem("tts_api_key") || ""; }
function setApiKey(k) { localStorage.setItem("tts_api_key", (k || "").trim()); }

async function api(path, options = {}) {
  const headers = new Headers(options.headers || {});
  const key = getApiKey();
  if (key) headers.set("X-API-Key", key);
  if (options.body && !(options.body instanceof FormData) && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }
  const res = await fetch(path, { ...options, headers });
  const ct = res.headers.get("content-type") || "";
  let data = null;
  if (res.status !== 204) {
    data = ct.includes("application/json") ? await res.json() : await res.blob();
  }
  if (!res.ok) {
    const msg = (data && (data.detail || data.error)) || `HTTP ${res.status}`;
    const err = new Error(msg);
    err.status = res.status;
    err.payload = data;
    throw err;
  }
  return data;
}

/* ---------------------------------------------------------------- */
/* UI 反馈                                                           */
/* ---------------------------------------------------------------- */
function setStatus(text, withTime = true) {
  $("#statusText").textContent = text;
  if (withTime) {
    $("#statusTime").textContent = new Date().toLocaleTimeString("zh-CN", { hour12: false });
  }
}

let toastTimer = null;
function toast(msg, type = "info", ms = 3600) {
  const t = $("#toast");
  t.textContent = msg;
  t.className = `toast ${type}`;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.add("hidden"), ms);
}

function setBusy(btn, on, label) {
  if (on) {
    btn.dataset.label = btn.querySelector(".btn-label")?.textContent || label || "";
    btn.classList.add("loading");
    btn.disabled = true;
    btn.querySelector(".btn-label").textContent = label || "处理中…";
  } else {
    btn.classList.remove("loading");
    btn.disabled = false;
    btn.querySelector(".btn-label").textContent = btn.dataset.label || "开始合成";
  }
}

/* ---------------------------------------------------------------- */
/* 拖拽上传                                                          */
/* ---------------------------------------------------------------- */
function bindDrop(zoneSel, inputSel, infoSel, onFile) {
  const zone = $(zoneSel);
  const input = $(inputSel);
  const info = $(infoSel);

  const showFile = (file) => {
    if (!file) return;
    const size = file.size > 1024 * 1024 ? (file.size / 1024 / 1024).toFixed(2) + " MB" : (file.size / 1024).toFixed(0) + " KB";
    info.textContent = `已选择：${file.name}（${size}）`;
    info.classList.remove("hidden");
    zone.querySelector(".drop-inner").style.display = "none";
    onFile && onFile(file);
  };

  zone.addEventListener("click", () => input.click());
  input.addEventListener("change", () => { showFile(input.files[0]); input.value = ""; });

  ["dragenter", "dragover"].forEach((ev) =>
    zone.addEventListener(ev, (e) => { e.preventDefault(); zone.classList.add("dragover"); }));
  ["dragleave", "drop"].forEach((ev) =>
    zone.addEventListener(ev, (e) => { e.preventDefault(); zone.classList.remove("dragover"); }));
  zone.addEventListener("drop", (e) => {
    const file = e.dataTransfer.files && e.dataTransfer.files[0];
    if (file) showFile(file);
  });
}

function getFile(infoSel) { return window["_file_" + infoSel.replace(/\W/g, "")] || null; }

/* ---------------------------------------------------------------- */
/* 模型信息与语种                                                    */
/* ---------------------------------------------------------------- */
function populateLang(sel, languages) {
  const keep = $(sel).value || "Auto";
  const opts = new Set(["Auto"]);
  (languages || []).forEach((l) => { if (String(l).trim()) opts.add(l); });
  $(sel).innerHTML = "";
  opts.forEach((l) => {
    const o = document.createElement("option");
    o.value = l;
    o.textContent = l;
    $(sel).appendChild(o);
  });
  $(sel).value = opts.has(keep) ? keep : "Auto";
}

async function loadModelInfo() {
  const chip = $("#modelChip");
  const chipText = $("#modelChipText");
  try {
    const info = await api("/api/v1/models/info");
    populateLang("#synthLang", info.languages);
    populateLang("#cLang", info.languages);
    if (info.ready) {
      state.ready = true;
      chip.className = "chip ready";
      const cap = info.cuda_capability ? `sm_${info.cuda_capability[0]}${info.cuda_capability[1]}` : "CPU";
      chipText.textContent = `${info.name} · ${info.device} · ${info.dtype} · ${info.attn_implementation || "eager"} · ${cap}`;
    } else {
      chip.className = "chip error";
      chipText.textContent = "模型未加载";
    }
  } catch (e) {
    chip.className = "chip error";
    chipText.textContent = e.status === 401 ? "需要 API Key" : "模型信息获取失败";
    if (e.status === 401) toast("服务启用了鉴权，请点击右上角 API Key 配置密钥。", "warn", 5000);
  }
}

/* ---------------------------------------------------------------- */
/* 音色库                                                            */
/* ---------------------------------------------------------------- */
function fmtDur(s) { return (s >= 60) ? `${Math.floor(s / 60)}m${(s % 60).toFixed(0)}s` : `${Number(s).toFixed(1)}s`; }
function fmtTime(ts) {
  const d = new Date(ts * 1000);
  const p = (n) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}`;
}

function avatarText(name) {
  const m = name.trim().match(/^[\u4e00-\u9fff]/);
  return m ? m[0] : (name.trim()[0] || "?").toUpperCase();
}

async function loadVoices() {
  const list = $("#voiceList");
  try {
    const res = await api("/api/v1/voices");
    state.voices = res.voices || [];
    renderVoices();
    $("#voiceCount").textContent = state.voices.length;
  } catch (e) {
    list.innerHTML = `<div class="empty-tip">音色库加载失败：${escapeHtml(e.message)}</div>`;
    toast(`音色库加载失败：${e.message}`, "err");
  }
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function renderVoices() {
  const list = $("#voiceList");
  const kw = ($("#searchInput").value || "").trim().toLowerCase();
  const items = state.voices.filter((v) => !kw || v.name.toLowerCase().includes(kw));
  const count = state.voices.length;

  const el = document.createElement("div");
  el.className = "voice-list";
  el.id = "voiceList";

  if (!count) {
    el.innerHTML = `<div class="empty-tip">还没有音色，点击上方「上传新音色」添加。</div>`;
  } else if (!items.length) {
    el.innerHTML = `<div class="empty-tip">没有匹配“${escapeHtml(kw)}”的音色。</div>`;
  }
  items.forEach((v) => {
    const item = document.createElement("div");
    item.className = "voice-item" + (v.id === state.selectedVoiceId ? " selected" : "");
    item.innerHTML = `
      <div class="voice-avatar">${escapeHtml(avatarText(v.name))}</div>
      <div class="voice-body">
        <div class="voice-name" title="${escapeHtml(v.name)}">${escapeHtml(v.name)}</div>
        <div class="voice-sub">
          <span class="tag ${v.mode === "icl" ? "tag-icl" : "tag-xvec"}">${v.mode === "icl" ? "ICL 参考文本" : "x-vector"}</span>
          <span>${fmtDur(v.duration_seconds)}</span>
          <span>${fmtTime(v.created_at)}</span>
        </div>
      </div>
      <button class="voice-del" title="删除音色" aria-label="删除音色" data-id="${v.id}">
        <svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M3 6h18M8 6V4a1 1 0 0 1 1-1h6a1 1 0 0 1 1 1v2M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/></svg>
      </button>`;
    item.querySelector(".voice-avatar").style.background =
      `linear-gradient(120deg, ${hashColor(v.id)}, ${hashColor(v.id + "x")})`;
    item.addEventListener("click", (e) => {
      if (e.target.closest(".voice-del")) return;
      selectVoice(v.id);
    });
    item.querySelector(".voice-del").addEventListener("click", async (e) => {
      e.stopPropagation();
      await deleteVoice(v.id);
    });
    el.appendChild(item);
  });

  list.replaceWith(el);
  syncSelectOptions();
  updateSelectedInfo();
}

function hashColor(salt) {
  let h = 0;
  for (let i = 0; i < salt.length; i++) h = (h * 31 + salt.charCodeAt(i)) | 0;
  const colors = ["#4F46E5", "#7C3AED", "#06B6D4", "#3B82F6", "#8B5CF6", "#0EA5E9", "#6366F1", "#14B8A6"];
  return colors[Math.abs(h) % colors.length];
}

function syncSelectOptions() {
  const sel = $("#voiceSelect");
  const current = state.selectedVoiceId || sel.value;
  sel.innerHTML = "";
  if (!state.voices.length) {
    const o = document.createElement("option");
    o.value = "";
    o.textContent = "（暂无音色，请先上传）";
    sel.appendChild(o);
    sel.disabled = true;
    return;
  }
  sel.disabled = false;
  state.voices.forEach((v) => {
    const o = document.createElement("option");
    o.value = v.id;
    o.textContent = `${v.name}（${v.mode === "icl" ? "参考文本" : "x-vector"} · ${fmtDur(v.duration_seconds)}）`;
    sel.appendChild(o);
  });
  if (current && state.voices.some((v) => v.id === current)) sel.value = current;
  else if (state.voices[0]) selectVoice(state.voices[0].id);
}

function selectVoice(id) {
  state.selectedVoiceId = id;
  renderVoices();
  $("#synthLang").focus && $("#synthLang").blur();
}

function updateSelectedInfo() {
  const v = state.voices.find((x) => x.id === state.selectedVoiceId);
  $("#savedSelectedInfo").textContent = v
    ? `已选：${v.name} · ${v.mode === "icl" ? "ICL" : "x-vector"} · ${fmtDur(v.duration_seconds)}`
    : "未选择";
}

async function deleteVoice(id) {
  const v = state.voices.find((x) => x.id === id);
  if (!v) return;
  if (!window.confirm(`确定删除音色「${v.name}」吗？此操作不可恢复。`)) return;
  try {
    await api(`/api/v1/voices/${id}`, { method: "DELETE" });
    if (state.selectedVoiceId === id) state.selectedVoiceId = null;
    toast(`已删除音色「${v.name}」`, "ok");
    await loadVoices();
  } catch (e) {
    toast(`删除失败：${e.message}`, "err");
  }
}

/* ---------------------------------------------------------------- */
/* 上传保存音色                                                      */
/* ---------------------------------------------------------------- */
function openModal() { $("#voiceModal").classList.remove("hidden"); resetUploadForm(); }
function closeModal() { $("#voiceModal").classList.add("hidden"); }

function resetUploadForm() {
  $("#vName").value = "";
  $("#vRefText").value = "";
  $("#vXvec").checked = false;
  const drop = $("#vDrop");
  drop.querySelector(".drop-inner").style.display = "";
  $("#vFileInfo").classList.add("hidden");
  window["_file_vFileInput"] = null;
}

async function saveVoice() {
  const name = $("#vName").value.trim();
  const file = window["_file_vFileInput"];
  const refText = $("#vRefText").value.trim();
  const xvec = $("#vXvec").checked;
  if (!name) return toast("请填写音色名称", "warn");
  if (!file) return toast("请选择参考音频文件", "warn");
  if (!xvec && !refText) return toast("ICL 模式必须填写参考音频文本；如需免文本请勾选「仅使用说话人向量」", "warn");
  if (!state.ready) return toast("模型未就绪，无法提取声纹", "err");

  const fd = new FormData();
  fd.append("file", file, file.name);
  fd.append("name", name);
  if (refText) fd.append("ref_text", refText);
  fd.append("x_vector_only", String(xvec));

  const btn = $("#saveVoiceBtn");
  setBusy(btn, true, "提取声纹中…");
  try {
    const meta = await api("/api/v1/voices", { method: "POST", body: fd });
    toast(`音色「${meta.name}」保存成功（voice_id: ${meta.id}）`, "ok", 5000);
    closeModal();
    await loadVoices();
    selectVoice(meta.id);
    setStatus(`已保存音色 ${meta.name}`);
  } catch (e) {
    toast(`保存失败：${e.message}`, "err", 5000);
  } finally {
    setBusy(btn, false);
  }
}

/* ---------------------------------------------------------------- */
/* 合成（已保存音色 / 一次性克隆）                                     */
/* ---------------------------------------------------------------- */
function renderResult(url, sr) {
  if (state.resultUrl) URL.revokeObjectURL(state.resultUrl);
  state.resultUrl = url;
  $("#resultAudio").src = url;
  const dl = $("#downloadLink");
  dl.href = url;
  dl.download = `tts_${Date.now()}.wav`;
  $("#resultMeta").textContent = sr ? `采样率 ${sr} Hz` : "";
  $("#resultCard").classList.remove("hidden");
  setStatus("合成完成，可试听或下载");
}

function collectAdvanced() {
  const g = {};
  const map = [
    ["#pTemp", "temperature", parseFloat],
    ["#pTopP", "top_p", parseFloat],
    ["#pTopK", "top_k", parseInt],
    ["#pRep", "repetition_penalty", parseFloat],
    ["#pMax", "max_new_tokens", parseInt],
  ];
  map.forEach(([sel, key, fn]) => {
    const raw = $(sel).value;
    if (raw !== "" && raw != null) {
      const val = fn(raw);
      if (!Number.isNaN(val)) g[key] = val;
    }
  });
  return g;
}

async function synthSaved() {
  const voiceId = state.selectedVoiceId || $("#voiceSelect").value;
  const text = $("#synthText").value.trim();
  if (!voiceId) return toast("请先选择音色", "warn");
  if (!text) return toast("请输入待合成文本", "warn");
  if (!state.ready) return toast("模型未就绪", "err");

  const body = {
    voice_id: voiceId,
    text,
    language: $("#synthLang").value || "Auto",
    ...collectAdvanced(),
  };
  const btn = $("#synthBtn");
  setBusy(btn, true, "合成中（耗时与文本长度相关）…");
  setStatus("正在合成…");
  try {
    const blob = await api("/api/v1/tts", { method: "POST", body: JSON.stringify(body) });
    renderResult(URL.createObjectURL(blob), blob.type === "audio/wav" ? null : null);
  } catch (e) {
    toast(`合成失败：${e.message}`, "err", 6000);
    setStatus(`合成失败：${e.message}`, false);
  } finally {
    setBusy(btn, false);
  }
}

async function synthClone() {
  const file = getFile("#cDrop");
  const text = $("#cSynthText").value.trim();
  const refText = $("#cRefText").value.trim();
  const xvec = $("#cXvec").checked;
  if (!file) return toast("请选择参考音频", "warn");
  if (!text) return toast("请输入待合成文本", "warn");
  if (!xvec && !refText) return toast("请填写参考音频文本，或勾选「仅使用说话人向量」", "warn");
  if (!state.ready) return toast("模型未就绪", "err");

  const fd = new FormData();
  fd.append("file", file, file.name);
  fd.append("text", text);
  fd.append("language", $("#cLang").value || "Auto");
  fd.append("x_vector_only", String(xvec));
  if (refText) fd.append("ref_text", refText);

  const btn = $("#cSynthBtn");
  setBusy(btn, true, "克隆合成中…");
  setStatus("正在克隆合成…");
  try {
    const blob = await api("/api/v1/tts/clone", { method: "POST", body: fd });
    renderResult(URL.createObjectURL(blob));
  } catch (e) {
    toast(`合成失败：${e.message}`, "err", 6000);
    setStatus(`合成失败：${e.message}`, false);
  } finally {
    setBusy(btn, false);
  }
}

/* ---------------------------------------------------------------- */
/* 事件绑定与初始化                                                  */
/* ---------------------------------------------------------------- */
function bindEvents() {
  // tabs
  $$(".tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      $$(".tab").forEach((t) => t.classList.toggle("active", t === tab));
      $$(".panel").forEach((p) => p.classList.toggle("active", p.id === tab.dataset.tab));
    });
  });

  // API Key 弹层
  $("#apiKeyBtn").addEventListener("click", (e) => {
    e.stopPropagation();
    const pop = $("#apiKeyPop");
    pop.classList.toggle("hidden");
    if (!pop.classList.contains("hidden")) $("#apiKeyInput").value = getApiKey();
  });
  $("#apiKeyClose").addEventListener("click", () => $("#apiKeyPop").classList.add("hidden"));
  document.addEventListener("click", (e) => {
    if (!e.target.closest("#apiKeyPop") && !e.target.closest("#apiKeyBtn")) {
      $("#apiKeyPop").classList.add("hidden");
    }
  });
  $("#apiKeySave").addEventListener("click", () => {
    setApiKey($("#apiKeyInput").value);
    $("#apiKeyPop").classList.add("hidden");
    toast("API Key 已保存", "ok");
    loadModelInfo();
    loadVoices();
  });

  // 模态
  $("#newVoiceBtn").addEventListener("click", openModal);
  $("#modalClose").addEventListener("click", closeModal);
  $("#voiceModal").addEventListener("click", (e) => { if (e.target === e.currentTarget) closeModal(); });

  // 搜索
  $("#searchInput").addEventListener("input", renderVoices);

  // 音色选择
  $("#voiceSelect").addEventListener("change", (e) => {
    state.selectedVoiceId = e.target.value;
    renderVoices();
  });

  // 合成
  $("#synthBtn").addEventListener("click", synthSaved);
  $("#cSynthBtn").addEventListener("click", synthClone);

  // 一次性克隆 xvec 提示
  $("#cXvec").addEventListener("change", (e) => {
    $("#cRefText").disabled = e.target.checked;
  });
  $("#vXvec").addEventListener("change", (e) => {
    $("#vRefText").disabled = e.target.checked;
  });

  // 保存音色
  $("#saveVoiceBtn").addEventListener("click", saveVoice);

  // 上传
  bindDrop("#vDrop", "#vFileInput", "#vFileInfo", (f) => { window["_file_vFileInput"] = f; });
  bindDrop("#cDrop", "#cFileInput", "#cFileInfo", (f) => { window["_file_cDrop"] = f; });

  // 回车快捷：Ctrl/Cmd+Enter 触发合成
  document.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
      const active = document.querySelector(".panel.active");
      if (active && active.id === "panelSaved") synthSaved();
      else if (active && active.id === "panelTemp") synthClone();
    }
  });
}

async function init() {
  bindEvents();
  setStatus("正在连接服务…");
  await Promise.all([loadModelInfo(), loadVoices()]);
  setStatus("就绪");
}

document.addEventListener("DOMContentLoaded", init);
