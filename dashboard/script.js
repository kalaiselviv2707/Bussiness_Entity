"use strict";
const $ = id => document.getElementById(id);
const fmt = n => typeof n === "number" ? n.toLocaleString() : String(n ?? "-");
const pct4 = n => typeof n === "number" ? n.toFixed(4) : "-";
const pct1 = n => typeof n === "number" ? (n * 100).toFixed(1) + "%" : "-";
const esc = s => String(s).replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));

const C = { accent:"#38bdf8", indigo:"#6366f1", good:"#34d399", warn:"#fbbf24", bad:"#f87171", purple:"#a78bfa", orange:"#fb923c" };
const PAL = [C.accent, C.indigo, C.good, C.warn, C.purple, C.orange, C.bad, "#f472b6"];
const GRID = { grid:{ color:"#1c3058" }, ticks:{ color:"#6b82a8" } };

const charts = {};
const state1 = { status:"all", q:"", page:1, pages:1 };
const state2 = { status:"all", q:"", page:1, pages:1 };

Chart.defaults.color = "#6b82a8";
Chart.defaults.font.family = "'Segoe UI', system-ui, sans-serif";

// ── API ──────────────────────────────────────────────────────
async function api(path) {
  const r = await fetch(path);
  let b = null;
  try { b = await r.json(); } catch(_) {}
  if (!r.ok) throw new Error((b && b.error) || `HTTP ${r.status}`);
  return b;
}

// ── CHART HELPER ─────────────────────────────────────────────
function draw(id, cfg) {
  if (!$(id)) return;
  if (charts[id]) charts[id].destroy();
  charts[id] = new Chart($(id), cfg);
}

// ── ANIMATED COUNTER ─────────────────────────────────────────
function countUp(el, target, ms = 750) {
  const t0 = performance.now();
  const isInt = Number.isInteger(target);
  (function step(now) {
    const p = Math.min((now - t0) / ms, 1);
    const ease = 1 - Math.pow(1 - p, 3);
    const v = target * ease;
    el.textContent = isInt ? Math.round(v).toLocaleString() : v.toFixed(2);
    if (p < 1) requestAnimationFrame(step);
    else el.textContent = isInt ? target.toLocaleString() : target.toFixed(2);
  })(t0);
}

// ── KPI CARDS ────────────────────────────────────────────────
const KPI_DEFS = [
  { id:"kc0", label:"Source 1 Records",      key:"total_source1",           icon:"&#11041;", grad:"#1e3a5f,#38bdf8", cls:"c-accent" },
  { id:"kc1", label:"Matched Entities",       key:"matched_entities",        icon:"&#10003;", grad:"#064e3b,#34d399", cls:"c-good"   },
  { id:"kc2", label:"Unmatched Entities",     key:"unmatched_entities",      icon:"&#9711;",  grad:"#451a03,#fbbf24", cls:"c-warn"   },
  { id:"kc3", label:"Candidate Pairs",        key:"candidate_pairs",         icon:"&#8983;",  grad:"#1e1b4b,#6366f1", cls:"c-indigo" },
  { id:"kc4", label:"Final Matches",          key:"final_matches",           icon:"&#9733;",  grad:"#064e3b,#10b981", cls:"c-good"   },
  { id:"kc5", label:"Avg Candidates / Entity",key:"avg_candidates_per_entity",icon:"&#8776;", grad:"#1e3a5f,#0ea5e9", cls:""         },
];

function renderKPIs(d) {
  KPI_DEFS.forEach(({ id, label, key, icon, grad, cls }) => {
    const card = $(id);
    if (!card) return;
    const val = d[key];
    card.innerHTML = `
      <div class="kpi-ico" style="background:linear-gradient(135deg,${grad})">${icon}</div>
      <div>
        <div class="kpi-lbl">${label}</div>
        <div class="kpi-val ${cls}" id="kv_${key}">0</div>
      </div>`;
    const el = $(`kv_${key}`);
    if (typeof val === "number") countUp(el, val);
    else el.textContent = val ?? "-";
  });
}

// ── OVERVIEW CHARTS ──────────────────────────────────────────
function renderOverview(d) {
  renderKPIs(d);

  // Donut
  const rate = d.total_source1 ? ((d.matched_entities / d.total_source1) * 100).toFixed(1) : 0;
  const dl = $("donutLabel");
  if (dl) dl.innerHTML = `<div class="dl-pct">${rate}%</div><div class="dl-sub">Match Rate</div>`;
  draw("matchDonut", {
    type:"doughnut",
    data:{ labels:["Matched","Unmatched"],
      datasets:[{ data:[d.matched_entities, d.unmatched_entities],
        backgroundColor:[C.good, C.warn], borderColor:"#0f1e35", borderWidth:4, hoverOffset:6 }] },
    options:{ maintainAspectRatio:false, cutout:"70%",
      plugins:{ legend:{ position:"bottom", labels:{ padding:14, usePointStyle:true } },
        tooltip:{ callbacks:{ label: ctx => ` ${ctx.label}: ${ctx.parsed.toLocaleString()}` } } } }
  });

  // Funnel
  draw("funnelChart", {
    type:"bar",
    data:{ labels:["Source 1 Records","Candidate Pairs","Final Matches"],
      datasets:[{ data:[d.total_source1, d.candidate_pairs, d.final_matches],
        backgroundColor:[C.accent, C.indigo, C.good], borderRadius:7, borderSkipped:false }] },
    options:{ maintainAspectRatio:false, indexAxis:"y",
      plugins:{ legend:{ display:false } },
      scales:{ x:{ ...GRID, beginAtZero:true }, y:{ ...GRID } } }
  });

  // Match dist
  const dist = d.matches_per_entity || {};
  const dlbls = Object.keys(dist).map(k => k === "5" ? "5+ matches" : k === "0" ? "0 matches" : k + (k==="1"?" match":" matches"));
  draw("matchDistChart", {
    type:"bar",
    data:{ labels:dlbls,
      datasets:[{ data:Object.values(dist), backgroundColor:PAL, borderRadius:6, borderSkipped:false }] },
    options:{ maintainAspectRatio:false,
      plugins:{ legend:{ display:false } },
      scales:{ x:{ ...GRID }, y:{ ...GRID, beginAtZero:true } } }
  });

  // Country
  const countries = Object.entries(d.by_country || {});
  const cc = $("countryCard");
  if (cc) cc.hidden = countries.length === 0;
  if (countries.length) {
    draw("countryChart", {
      type:"bar",
      data:{ labels:countries.map(([c]) => c || "Unknown"),
        datasets:[
          { label:"Matched",   data:countries.map(([,v]) => v.matched),           backgroundColor:C.good, borderRadius:4 },
          { label:"Unmatched", data:countries.map(([,v]) => v.total - v.matched), backgroundColor:C.warn, borderRadius:4 }
        ] },
      options:{ maintainAspectRatio:false,
        plugins:{ legend:{ position:"bottom", labels:{ usePointStyle:true } } },
        scales:{ x:{ ...GRID, stacked:true }, y:{ ...GRID, stacked:true, beginAtZero:true } } }
    });
  }
}

// ── MODEL ────────────────────────────────────────────────────
function renderModel(m) {
  const set = (id, v) => { const e = $(id); if (e) e.textContent = v; };
  set("mPrecision",  pct4(m.precision));
  set("mRecall",     pct4(m.recall));
  set("mF05",        pct4(m.f0_5));
  set("mThreshold",  m.threshold != null ? m.threshold.toFixed(2) : "-");
  set("modelBadge",  m.model || "");
  set("confTP",      fmt(m.tp));
  set("confFP",      fmt(m.fp));
  set("confFN",      fmt(m.fn));

  const bar = (id, v) => { const e = $(id); if (e) e.style.width = (Math.min(v||0,1)*100)+"%"; };
  bar("barPrecision", m.precision);
  bar("barRecall",    m.recall);
  bar("barF05",       m.f0_5);

  const t = m.training || {};
  set("modelNote", `Validation: ${t.validation_split||"n/a"} · TP ${fmt(m.tp)} / FP ${fmt(m.fp)} / FN ${fmt(m.fn)}` +
    (t.blocking_recall != null ? ` · Blocking recall ${(t.blocking_recall*100).toFixed(2)}%` : ""));

  // Radar
  draw("radarChart", {
    type:"radar",
    data:{ labels:["Precision","Recall","F0.5","Blocking Recall","Threshold Conf."],
      datasets:[{ label:m.model||"Model",
        data:[m.precision||0, m.recall||0, m.f0_5||0, t.blocking_recall||0,
              m.threshold!=null ? 1-Math.abs(m.threshold-0.5)*2 : 0],
        backgroundColor:"rgba(99,102,241,.18)", borderColor:C.indigo,
        pointBackgroundColor:C.accent, pointRadius:5, borderWidth:2 }] },
    options:{ maintainAspectRatio:false,
      scales:{ r:{ min:0, max:1, grid:{ color:"#1c3058" }, ticks:{ display:false },
        pointLabels:{ color:"#6b82a8", font:{ size:11 } }, angleLines:{ color:"#1c3058" } } },
      plugins:{ legend:{ display:false } } }
  });

  // Comparison table
  const comp = m.model_comparison || [];
  const tb = $("modelCompBody");
  if (comp.length && tb) {
    tb.innerHTML = comp.map(r => `<tr>
      <td><b>${esc(r.model)}</b></td>
      <td><b style="color:${C.good}">${pct4(r.f0_5)}</b></td>
      <td>${pct4(r.precision)}</td><td>${pct4(r.recall)}</td>
      <td>${r.threshold!=null?r.threshold.toFixed(2):"-"}</td>
      <td style="color:${C.good}">${fmt(r.tp)}</td>
      <td style="color:${C.warn}">${fmt(r.fp)}</td>
      <td style="color:${C.bad}">${fmt(r.fn)}</td>
      <td>${r.model===m.model?'<span class="pill pill-best">&#10003; Selected</span>':""}</td>
    </tr>`).join("");

    draw("modelCompChart", {
      type:"bar",
      data:{ labels:comp.map(r => r.model),
        datasets:[
          { label:"F0.5",      data:comp.map(r=>r.f0_5),      backgroundColor:C.good,   borderRadius:6 },
          { label:"Precision", data:comp.map(r=>r.precision), backgroundColor:C.accent,  borderRadius:6 },
          { label:"Recall",    data:comp.map(r=>r.recall),    backgroundColor:C.indigo,  borderRadius:6 }
        ] },
      options:{ maintainAspectRatio:false,
        plugins:{ legend:{ position:"bottom", labels:{ usePointStyle:true } } },
        scales:{ x:{ ...GRID }, y:{ ...GRID, min:0, max:1 } } }
    });
  } else {
    const mc = $("modelCompCard");
    if (mc) mc.hidden = true;
  }
}

// ── PIPELINE ─────────────────────────────────────────────────
function renderPipeline(m) {
  const t = m.training || {};
  const set = (id, v) => { const e = $(id); if (e) e.textContent = v; };
  set("pipeModelName",      m.model || "Classifier");
  set("pipeThreshold",      m.threshold != null ? "thr = " + m.threshold.toFixed(2) : "Optimised");
  set("pipeBlockingRecall", t.blocking_recall != null ? (t.blocking_recall*100).toFixed(2)+"%" : "-");
  set("pipeCandPairs",      t.candidate_pairs  != null ? t.candidate_pairs.toLocaleString()  : "-");
  set("pipeTrainPairs",     t.training_pairs   != null ? t.training_pairs.toLocaleString()   : "-");
  set("pipePosPairs",       t.positive_pairs   != null ? t.positive_pairs.toLocaleString()   : "-");
  set("pipeFeatures",       t.feature_count    != null ? t.feature_count                     : "-");
  set("pipeLimit",          t.record_limit     != null ? t.record_limit.toLocaleString()     : "-");
}

// ── CANDIDATES ───────────────────────────────────────────────
function renderCandidates(c) {
  const set = (id, v) => { const e = $(id); if (e) e.textContent = v; };
  set("cAvg",      c.average);
  set("cMax",      fmt(c.max));
  set("cTotal",    fmt(c.total_candidate_pairs));
  set("cFinal",    fmt(c.final_matches));
  set("cRejected", fmt(c.rejected_candidates));
  set("cNone",     fmt(c.entities_without_candidates));

  const rate = c.total_candidate_pairs ? c.final_matches / c.total_candidate_pairs : 0;
  const rp = $("matchRatePct"); if (rp) rp.textContent = pct1(rate);
  const rb = $("matchRateBar"); if (rb) rb.style.width = (rate*100)+"%";

  draw("candDistChart", {
    type:"bar",
    data:{ labels:Object.keys(c.distribution||{}),
      datasets:[{ label:"Entities", data:Object.values(c.distribution||{}),
        backgroundColor:PAL, borderRadius:6, borderSkipped:false }] },
    options:{ maintainAspectRatio:false,
      plugins:{ legend:{ display:false },
        title:{ display:true, text:"Candidates per Source 1 entity", color:"#6b82a8", font:{ size:11 } } },
      scales:{ x:{ ...GRID }, y:{ ...GRID, beginAtZero:true } } }
  });

  const top = c.top_entities || [];
  const maxC = top.length ? top[0].candidate_count : 1;
  const tb = $("topEntitiesBody");
  if (tb) tb.innerHTML = top.map((e,i) => `<tr>
    <td class="muted">${i+1}</td>
    <td class="mono">${esc(e.source1_entity_id)}</td>
    <td><b style="color:var(--accent)">${fmt(e.candidate_count)}</b></td>
    <td><div class="mbar-wrap"><div class="mbar-fill" style="width:${(e.candidate_count/maxC*100).toFixed(1)}%"></div></div></td>
  </tr>`).join("");
}

// ── FEATURE IMPORTANCE ───────────────────────────────────────
async function renderFeatures() {
  const fc = $("featCard");
  try {
    const d = await api("/api/feature-importance");
    if (d.error) { if (fc) fc.innerHTML = `<div class="card-hdr">Feature Importance</div><p class="muted" style="padding:12px">${esc(d.error)}</p>`; return; }
    draw("featChart", {
      type:"bar",
      data:{ labels:d.features,
        datasets:[{ label:"Importance", data:d.importances,
          backgroundColor:d.features.map((_,i) => PAL[i % PAL.length]),
          borderRadius:5, borderSkipped:false }] },
      options:{ maintainAspectRatio:false, indexAxis:"y",
        plugins:{ legend:{ display:false } },
        scales:{ x:{ ...GRID, beginAtZero:true }, y:{ ...GRID, ticks:{ font:{ size:11 } } } } }
    });
  } catch(e) {
    if (fc) fc.innerHTML = `<div class="card-hdr">Feature Importance</div><p class="muted" style="padding:12px">Could not load feature importance: ${esc(e.message)}</p>`;
  }
}

// ── SYSTEM STATUS ────────────────────────────────────────────
async function renderSystem() {
  const grid = $("sysGrid");
  if (!grid) return;
  try {
    const d = await api("/api/system-status");
    const icons = { "matching_results.tsv":"&#128202;", "candidate_pairs.tsv":"&#128203;",
                    "model_metrics.json":"&#128200;", "model.pkl":"&#129302;" };
    grid.innerHTML = (d.components || []).map(c => `
      <div class="sys-card ${c.ok?"ok":"err"}">
        <div class="sys-icon ${c.ok?"ok":"err"}">${icons[c.name]||"&#128196;"}</div>
        <div>
          <div class="sys-name">${esc(c.name)}</div>
          <div class="sys-detail">${esc(c.detail)}</div>
          <div class="sys-status">${c.ok?"&#9679; Online":"&#9679; Offline"}</div>
        </div>
      </div>`).join("");

    // Health pills in topbar
    const hp = $("healthPills");
    if (hp) hp.innerHTML = (d.components||[]).map(c =>
      `<div class="hp ${c.ok?"ok":"err"}"><span class="hp-dot"></span>${esc(c.name.replace(/\.(tsv|json|pkl)$/,""))}</div>`
    ).join("");
  } catch(e) {
    grid.innerHTML = `<div class="sys-card err"><div class="sys-icon err">&#9888;</div><div><div class="sys-name">Status unavailable</div><div class="sys-detail">${esc(e.message)}</div></div></div>`;
  }
}

// ── TABLE (shared renderer) ───────────────────────────────────
function makeTableRow(r, offset, i) {
  const ids = r.matched_entity_ids.length
    ? r.matched_entity_ids.map(id => `<span style="display:inline-block;margin:1px 3px 1px 0;padding:1px 7px;background:rgba(52,211,153,.1);border-radius:4px;font-size:10px;color:var(--good)">${esc(id)}</span>`).join("")
    : '<span class="muted">—</span>';
  return `<tr>
    <td class="muted">${offset+i+1}</td>
    <td class="mono">${esc(r.source1_entity_id)}</td>
    <td>${ids}</td>
    <td>${fmt(r.candidate_count)}</td>
    <td>${fmt(r.match_count)}</td>
    <td><span class="pill ${r.status==="Matched"?"pill-m":"pill-u"}">${r.status}</span></td>
  </tr>`;
}

function renderPageNums(container, current, total, onPage) {
  const pages = [];
  if (total <= 7) { for (let i=1;i<=total;i++) pages.push(i); }
  else {
    pages.push(1);
    if (current > 3) pages.push("…");
    for (let i=Math.max(2,current-1);i<=Math.min(total-1,current+1);i++) pages.push(i);
    if (current < total-2) pages.push("…");
    pages.push(total);
  }
  container.innerHTML = pages.map(p => p==="…"
    ? `<span style="color:var(--muted);padding:0 3px">…</span>`
    : `<button class="page-btn ${p===current?"current":""}" data-page="${p}">${p}</button>`
  ).join("");
  container.querySelectorAll(".page-btn").forEach(b =>
    b.addEventListener("click", () => onPage(+b.dataset.page))
  );
}

async function loadTable(tbodyId, prevId, nextId, pnId, rcId, st) {
  const tbody = $(tbodyId);
  if (!tbody) return;
  tbody.innerHTML = `<tr><td colspan="6" class="tc muted">Loading&hellip;</td></tr>`;
  const params = new URLSearchParams({ status:st.status, page:st.page, page_size:25 });
  if (st.q) params.set("q", st.q);
  try {
    const data = await api("/api/matches?" + params);
    st.pages = data.pages;
    const offset = (data.page-1)*25;
    tbody.innerHTML = data.rows.length
      ? data.rows.map((r,i) => makeTableRow(r, offset, i)).join("")
      : `<tr><td colspan="6" class="tc muted">No rows match the current filter.</td></tr>`;
    const rc = $(rcId); if (rc) rc.textContent = `${data.total.toLocaleString()} rows`;
    const pv = $(prevId); if (pv) pv.disabled = data.page <= 1;
    const nx = $(nextId); if (nx) nx.disabled = data.page >= data.pages;
    const pn = $(pnId);
    if (pn) renderPageNums(pn, data.page, data.pages, p => { st.page=p; loadTable(tbodyId,prevId,nextId,pnId,rcId,st); });
  } catch(e) {
    tbody.innerHTML = `<tr><td colspan="6" class="tc muted">${esc(e.message)}</td></tr>`;
  }
}

// ── STATUS ───────────────────────────────────────────────────
function setStatus(ok, msg) {
  const d = $("statusDot"); if (d) d.className = "dot " + (ok?"ok":"err");
  const t = $("statusText"); if (t) t.textContent = msg;
  const lb = $("liveBadge"); if (lb) lb.style.opacity = ok ? "1" : "0.3";
}
function showError(msg) {
  const b = $("errorBanner");
  if (!b) return;
  b.hidden = !msg;
  b.textContent = msg || "";
}

// ── LOAD ALL ─────────────────────────────────────────────────
async function loadAll() {
  setStatus(false, "Loading…");
  const [dash, cand, metrics] = await Promise.allSettled([
    api("/api/dashboard"), api("/api/candidates"), api("/api/model-metrics")
  ]);
  const errs = [];
  if (dash.status    === "fulfilled") renderOverview(dash.value);
  else errs.push(dash.reason.message);
  if (cand.status    === "fulfilled") renderCandidates(cand.value);
  else errs.push(cand.reason.message);
  if (metrics.status === "fulfilled") { renderModel(metrics.value); renderPipeline(metrics.value); }
  else errs.push(metrics.reason.message);

  showError(errs.length ? [...new Set(errs)].join(" | ") : "");
  setStatus(!errs.length, errs.length ? "Data unavailable" : "Live data loaded");

  await Promise.all([
    loadTable("tbody",  "prevBtn",  "nextBtn",  "pageNumbers",  "resultCount",  state1),
    loadTable("tbody2", "prevBtn2", "nextBtn2", "pageNumbers2", "resultCount2", state2),
    renderFeatures(),
    renderSystem()
  ]);
}

// ── SEARCH / FILTER WIRING ───────────────────────────────────
function wireSearch(inputId, filtersId, st, reload) {
  let timer;
  const inp = $(inputId);
  if (inp) inp.addEventListener("input", e => {
    clearTimeout(timer);
    timer = setTimeout(() => { st.q = e.target.value.trim(); st.page = 1; reload(); }, 280);
  });
  const fg = $(filtersId);
  if (fg) fg.addEventListener("click", e => {
    const btn = e.target.closest(".seg-btn");
    if (!btn) return;
    fg.querySelectorAll(".seg-btn").forEach(b => b.classList.toggle("active", b===btn));
    st.status = btn.dataset.status; st.page = 1; reload();
  });
}

wireSearch("searchInput",  "filters",  state1, () => loadTable("tbody",  "prevBtn",  "nextBtn",  "pageNumbers",  "resultCount",  state1));
wireSearch("searchInput2", "filters2", state2, () => loadTable("tbody2", "prevBtn2", "nextBtn2", "pageNumbers2", "resultCount2", state2));

const pv = $("prevBtn");  if (pv) pv.addEventListener("click", () => { if (state1.page>1){state1.page--;loadTable("tbody","prevBtn","nextBtn","pageNumbers","resultCount",state1);} });
const nx = $("nextBtn");  if (nx) nx.addEventListener("click", () => { if (state1.page<state1.pages){state1.page++;loadTable("tbody","prevBtn","nextBtn","pageNumbers","resultCount",state1);} });
const pv2=$("prevBtn2"); if (pv2) pv2.addEventListener("click",() => { if (state2.page>1){state2.page--;loadTable("tbody2","prevBtn2","nextBtn2","pageNumbers2","resultCount2",state2);} });
const nx2=$("nextBtn2"); if (nx2) nx2.addEventListener("click",() => { if (state2.page<state2.pages){state2.page++;loadTable("tbody2","prevBtn2","nextBtn2","pageNumbers2","resultCount2",state2);} });

const rb = $("refreshBtn"); if (rb) rb.addEventListener("click", loadAll);
const mb = $("menuBtn");    if (mb) mb.addEventListener("click", () => $("sidebar").classList.toggle("open"));

// ── SCROLL SPY ───────────────────────────────────────────────
const navLinks = document.querySelectorAll(".nav-link");
const observer = new IntersectionObserver(entries => {
  entries.forEach(e => {
    if (e.isIntersecting)
      navLinks.forEach(a => a.classList.toggle("active", a.dataset.section === e.target.id));
  });
}, { threshold:0.25 });
document.querySelectorAll("section[id]").forEach(s => observer.observe(s));
navLinks.forEach(a => a.addEventListener("click", () => $("sidebar").classList.remove("open")));

// ── BOOT ─────────────────────────────────────────────────────
loadAll();
