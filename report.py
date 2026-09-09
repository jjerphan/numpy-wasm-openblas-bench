"""Emit self-contained HTML reports (no CDN) for NumPy wasm BLAS/LAPACK compares."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

MAPPING = [
    {"api": "np.dot(x, y)", "group": "l1", "blas": "dot / zdotu"},
    {"api": "np.vdot", "group": "l1", "blas": "dotc"},
    {"api": "np.inner", "group": "l1", "blas": "dot"},
    {"api": "np.linalg.vecdot", "group": "l1", "blas": "dot"},
    {"api": "np.linalg.norm(x, 2) / vector_norm", "group": "l1", "blas": "nrm2 (or sqrt(dot))"},
    {"api": "A @ x / x @ A", "group": "l2", "blas": "gemv"},
    {"api": "A @ B / np.matmul / np.dot(A, B)", "group": "l3", "blas": "gemm"},
    {"api": "F-contiguous A @ B", "group": "l3", "blas": "gemm"},
    {"api": "A @ A.conj().T", "group": "l3", "blas": "syrk / herk (or gemm)"},
    {"api": "batched a @ b", "group": "l3", "blas": "gemm"},
    {"api": "np.tensordot / multi_dot / matrix_power", "group": "l3", "blas": "gemm"},
    {"api": "np.linalg.solve / tensorsolve", "group": "lapack", "blas": "gesv"},
    {"api": "inv / tensorinv / det / slogdet", "group": "lapack", "blas": "getrf"},
    {"api": "cholesky", "group": "lapack", "blas": "potrf"},
    {"api": "eigh / eigvalsh", "group": "lapack", "blas": "syevd / heevd"},
    {"api": "eig / eigvals", "group": "lapack", "blas": "geev"},
    {"api": "svd / svdvals / svd_full / pinv / cond / matrix_rank / matrix_norm", "group": "lapack", "blas": "gesdd"},
    {"api": "qr", "group": "lapack", "blas": "geqrf + orgqr"},
    {"api": "lstsq", "group": "lapack", "blas": "gelsd"},
]

LABEL_NAMES = {
    "openblas": "emscripten-forge OpenBLAS 0.3.34",
    "ob034": "emscripten-forge OpenBLAS 0.3.34",
    "obdev": "emscripten-forge OpenBLAS 0.3.35",
    "pyodide": "Pyodide NumPy",
    "noblas": "emscripten-forge NumPy (no-BLAS)",
    "cf64": "conda-forge linux-64 OpenBLAS 0.3.34",
}


def write_report(
    path: Path,
    rows: list[dict],
    groups: list[dict],
    openblas_meta: dict,
    noblas_meta: dict,
    comparisons: list[dict] | None = None,
) -> None:
    def pretty(meta: dict, fallback: str) -> str:
        label = str(meta.get("label") or "")
        names = LABEL_NAMES
        return names.get(label, fallback)

    if comparisons is None:
        comparisons = [
            {
                "id": "main",
                "left_label": pretty(openblas_meta, "OpenBLAS numpy"),
                "right_label": pretty(noblas_meta, "no-BLAS numpy"),
                "openblas": openblas_meta,
                "noblas": noblas_meta,
                "groups": groups,
                "rows": rows,
            }
        ]
    payload = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "mapping": MAPPING,
        "comparisons": comparisons,
        "openblas": openblas_meta,
        "noblas": noblas_meta,
        "left_label": comparisons[0].get("left_label", "OpenBLAS numpy"),
        "right_label": comparisons[0].get("right_label", "no-BLAS numpy"),
        "groups": groups,
        "rows": rows,
    }
    def json_safe(obj):
        if isinstance(obj, dict):
            return {k: json_safe(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [json_safe(v) for v in obj]
        if isinstance(obj, float) and (obj != obj or obj in (float("inf"), float("-inf"))):
            return None
        return obj

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(TEMPLATE.replace("__DATA__", json.dumps(json_safe(payload))))


TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NumPy wasm BLAS/LAPACK</title>
<style>
:root {
  --bg: #f7f5f2; --fg: #1c1b19; --muted: #5c5852; --line: #d9d4cc; --card: #fff;
  --accent: #1f4e79; --openblas: #1f4e79; --noblas: #a15c2d;
  --speed-f64: #1f4e79; --speed-f32: #1f7a4d; --speed-c128: #6b3fa0; --speed-c64: #b45309;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #161513; --fg: #f0ece6; --muted: #b0aaa2; --line: #3a3733; --card: #221f1c;
    --accent: #8cb4d9; --openblas: #8cb4d9; --noblas: #e0a070;
    --speed-f64: #8cb4d9; --speed-f32: #6dcaa0; --speed-c128: #c4a0e0; --speed-c64: #e0a070;
  }
}
* { box-sizing: border-box; }
body { margin: 0; font: 15px/1.45 "Segoe UI", system-ui, sans-serif; color: var(--fg); background: var(--bg); }
header, main { max-width: 1100px; margin: 0 auto; padding: 1.25rem 1.5rem; }
header h1 { font-size: 1.55rem; margin: 0 0 0.35rem; font-weight: 650; }
header p { margin: 0; color: var(--muted); }
.caption { color: var(--muted); font-size: 0.9rem; margin: 0.4rem 0 1rem; }
nav { display: flex; flex-wrap: wrap; gap: 0.5rem 1rem; margin: 0.75rem 0 0; }
nav a { color: var(--accent); }
.grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
@media (max-width: 800px) { .grid, .charts { grid-template-columns: 1fr; } }
section { background: var(--card); border: 1px solid var(--line); border-radius: 8px; padding: 1rem 1.1rem 1.15rem; margin: 0 0 1rem; }
h2 { font-size: 1.05rem; margin: 0 0 0.75rem; }
h3 { font-size: 1rem; margin: 1.25rem 0 0.6rem; }
table { width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }
th, td { text-align: left; padding: 0.28rem 0.45rem; border-bottom: 1px solid var(--line); }
th { font-weight: 600; color: var(--muted); font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.03em; }
td.num, th.num { text-align: right; }
.oom { color: #a33; }
dl { display: grid; grid-template-columns: 10rem 1fr; gap: 0.15rem 0.75rem; margin: 0; }
dt { color: var(--muted); }
dd { margin: 0; word-break: break-all; }
.charts { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
.charts svg { width: 100%; height: auto; background: transparent; }
.legend { font-size: 0.85rem; color: var(--muted); margin: 0 0 0.4rem; }
.curve-legend { display: flex; flex-wrap: wrap; align-items: center; gap: 0.35rem 1.25rem; margin: 0.15rem 0 0.85rem; font-size: 0.85rem; line-height: 1; color: var(--muted); }
.curve-legend-item { display: inline-flex; flex: 0 0 auto; align-items: center; gap: 0.4rem; line-height: 1; }
.curve-legend-item svg { width: 28px; height: 10px; flex: 0 0 28px; display: block; }
details { margin-top: 0.75rem; }
details summary { cursor: pointer; color: var(--accent); }
.cmp { border-top: 2px solid var(--line); margin-top: 1.5rem; padding-top: 0.5rem; }
</style>
</head>
<body>
<header>
  <h1>NumPy wasm: BLAS / LAPACK</h1>
  <p id="generated"></p>
  <p class="caption">NumPy Python APIs that dispatch to CBLAS or LAPACK (not every CBLAS/LAPACK symbol). Speedup &gt; 1 means OpenBLAS is faster. Protocol: 5 warmup + 10 timed samples, median. Headless Chromium / V8. Charts are speedup vs n for every op.</p>
  <nav id="nav"></nav>
</header>
<main>
  <section>
    <h2>NumPy API to BLAS/LAPACK</h2>
    <p class="caption">NumPy does not wrap ger/trsm/symm/etc. This table is the callable surface measured here.</p>
    <table id="mapping"></table>
  </section>
  <div id="comparisons"></div>
</main>
<script type="application/json" id="data">__DATA__</script>
<script>
(function () {
  const data = JSON.parse(document.getElementById("data").textContent);
  document.getElementById("generated").textContent = "Generated " + data.generated;
  function esc(s) {
    return String(s).replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"}[c]));
  }
  function num(x) { const v = Number(x); return Number.isFinite(v) ? v : null; }

  const map = document.getElementById("mapping");
  map.innerHTML = "<thead><tr><th>NumPy API</th><th>Group</th><th>BLAS/LAPACK</th></tr></thead><tbody>" +
    (data.mapping || []).map(m => "<tr><td>" + esc(m.api) + "</td><td>" + esc(m.group) + "</td><td>" + esc(m.blas) + "</td></tr>").join("") +
    "</tbody>";

  function metaHtml(meta) {
    const keys = [
      ["numpy_version", "numpy"], ["blas_name", "BLAS name"], ["blas_found", "BLAS found"],
      ["blas_version", "BLAS version"], ["blas_openblas_configuration", "OpenBLAS config"],
      ["openblas_conda", "OpenBLAS conda"], ["openblas_channel", "OpenBLAS channel"],
      ["lapack_name", "LAPACK name"], ["lapack_found", "LAPACK found"], ["lapack_version", "LAPACK version"],
      ["chromium", "Chromium"], ["conda_env", "conda env"], ["pyodide_index", "Pyodide"],
      ["python", "Python"], ["pyodide_version", "Pyodide version"],
    ];
    return keys.filter(([k]) => meta[k] !== undefined && meta[k] !== null && meta[k] !== "")
      .map(([k, label]) => "<div><dt>" + esc(label) + "</dt><dd>" + esc(String(meta[k])) + "</dd></div>").join("");
  }

  function curveLegendHtml(items) {
    return items.map(s => {
      const dash = s.dash ? ' stroke-dasharray="5 3"' : "";
      return `<span class="curve-legend-item"><svg width="28" height="10" viewBox="0 0 28 10" aria-hidden="true"><line x1="1" y1="5" x2="27" y2="5" stroke="${s.color}" stroke-width="2.2" stroke-linecap="round"${dash}/></svg><span>${esc(s.label)}</span></span>`;
    }).join("");
  }

  function lineChart(title, yLabel, seriesList, opts) {
    opts = opts || {};
    const w = 520, h = 280, l = 52, r = 16, t = 28, b = 36;
    const pts = seriesList.flatMap(s => s.pts);
    if (!pts.length) return "<p class='caption'>No data for " + esc(title) + "</p>";
    const xs = pts.map(p => p.n), ys = pts.map(p => p.y);
    const minX = Math.min(...xs), maxX = Math.max(...xs);
    const minY = 0;
    let maxY = Math.max(...ys) * 1.08 || 1;
    if (opts.href != null) maxY = Math.max(maxY, opts.href * 1.08);
    const xlog = minX > 0 && maxX / minX > 4;
    function sx(n) {
      if (xlog) return l + (Math.log(n) - Math.log(minX)) / (Math.log(maxX) - Math.log(minX) || 1) * (w - l - r);
      return l + (n - minX) / (maxX - minX || 1) * (w - l - r);
    }
    function sy(y) { return t + (1 - (y - minY) / (maxY - minY || 1)) * (h - t - b); }
    const ticks = [...new Set(xs)].sort((a,b)=>a-b);
    const yticks = 4;
    let grid = "";
    for (let i = 0; i <= yticks; i++) {
      const yv = minY + (maxY - minY) * i / yticks;
      const y = sy(yv);
      grid += `<line x1="${l}" y1="${y}" x2="${w-r}" y2="${y}" stroke="currentColor" opacity="0.12"/>`;
      grid += `<text x="${l-6}" y="${y+4}" text-anchor="end" font-size="10" fill="currentColor" opacity="0.7">${yv.toPrecision(3)}</text>`;
    }
    if (opts.href != null && opts.href >= minY && opts.href <= maxY) {
      const y = sy(opts.href);
      grid += `<line x1="${l}" y1="${y}" x2="${w-r}" y2="${y}" stroke="currentColor" stroke-width="1.4" opacity="0.55"/>`;
      const nearTick = [...Array(yticks + 1)].some((_, i) => Math.abs(minY + (maxY - minY) * i / yticks - opts.href) < (maxY - minY) * 0.04);
      if (!nearTick) grid += `<text x="${l-6}" y="${y+4}" text-anchor="end" font-size="10" fill="currentColor" opacity="0.85">${opts.href}</text>`;
    }
    ticks.forEach(n => { grid += `<text x="${sx(n)}" y="${h-12}" text-anchor="middle" font-size="10" fill="currentColor" opacity="0.7">${n}</text>`; });
    const paths = seriesList.map(s => {
      if (!s.pts.length) return "";
      const d = s.pts.map((p,i) => (i?"L":"M") + sx(p.n).toFixed(1) + " " + sy(p.y).toFixed(1)).join(" ");
      const dash = s.dash ? ' stroke-dasharray="5 4"' : "";
      const dots = s.pts.map(p => `<circle cx="${sx(p.n).toFixed(1)}" cy="${sy(p.y).toFixed(1)}" r="2.4" fill="${s.color}"/>`).join("");
      return `<path d="${d}" fill="none" stroke="${s.color}" stroke-width="1.8"${dash}/>${dots}`;
    }).join("");
    return `<div><div class="legend">${esc(title)}</div>
      <svg viewBox="0 0 ${w} ${h}" role="img" aria-label="${esc(title)}">${grid}${paths}
        <text x="${w/2}" y="${h-2}" text-anchor="middle" font-size="11" fill="currentColor" opacity="0.8">n</text>
        <text transform="translate(12,${h/2}) rotate(-90)" text-anchor="middle" font-size="11" fill="currentColor" opacity="0.8">${esc(yLabel)}</text>
      </svg></div>`;
  }

  const titles = {l1: "Level 1", l2: "Level 2", l3: "Level 3", lapack: "LAPACK", control: "control"};
  const groupOrder = ["l1", "l2", "l3", "lapack", "control"];
  const dtypeColors = {
    float64: "var(--speed-f64)", float32: "var(--speed-f32)",
  };
  const nav = document.getElementById("nav");
  const host = document.getElementById("comparisons");
  const comparisons = data.comparisons || [];
  comparisons.forEach((cmp, idx) => {
    const id = cmp.id || ("cmp" + idx);
    const left = cmp.left_label || "OpenBLAS numpy";
    const right = cmp.right_label || "other";
    nav.insertAdjacentHTML("beforeend", `<a href="#${id}">${esc(left)} vs ${esc(right)}</a>`);
    const wrap = document.createElement("div");
    wrap.className = "cmp";
    wrap.id = id;
    wrap.innerHTML = `<h2>${esc(left)} vs ${esc(right)}</h2>
      <p class="caption">Speedup = time(${esc(right)}) / time(${esc(left)}).</p>
      <div class="grid">
        <section><h2>${esc(left)}</h2><dl class="meta-left"></dl></section>
        <section><h2>${esc(right)}</h2><dl class="meta-right"></dl></section>
      </div>
      <section><h2>Geomean speedup by group</h2><table class="summary"></table></section>
      <section><h2>Speedup vs n</h2>
        <p class="caption">One chart per NumPy API. Speedup &gt; 1 (above the baseline) means OpenBLAS is faster.</p>
        <div class="curve-legend speedup-legend"></div>
        <div class="speedup-by-group"></div>
      </section>
      <section><h2>Per-op results</h2><div class="tables"></div></section>`;
    wrap.querySelector(".meta-left").innerHTML = metaHtml(cmp.openblas || {});
    wrap.querySelector(".meta-right").innerHTML = metaHtml(cmp.noblas || {});
    wrap.querySelector(".summary").innerHTML = "<thead><tr><th>Group</th><th class='num'>n</th><th class='num'>Geomean speedup</th></tr></thead><tbody>" +
      (cmp.groups || []).map(g => {
        const v = Number(g.geomean_speedup);
        return "<tr><td>" + esc(g.group) + "</td><td class='num'>" + g.n + "</td><td class='num'><strong>" +
          (Number.isFinite(v) ? v.toFixed(3) : "—") + "</strong></td></tr>";
      }).join("") + "</tbody>";

    wrap.querySelector(".speedup-legend").innerHTML = curveLegendHtml(
      Object.entries(dtypeColors).map(([k, c]) => ({label: k, color: c, dash: false}))
    );

    const rows = cmp.rows || [];
    const byOp = {};
    rows.forEach(r => { (byOp[r.op] ||= []).push(r); });
    const opsByGroup = {};
    Object.keys(byOp).forEach(op => {
      const g = byOp[op][0].group || "other";
      (opsByGroup[g] ||= []).push(op);
    });
    const speedHost = wrap.querySelector(".speedup-by-group");
    const groupsSeen = [...groupOrder.filter(g => opsByGroup[g]), ...Object.keys(opsByGroup).filter(g => !groupOrder.includes(g)).sort()];
    groupsSeen.forEach(g => {
      const ops = (opsByGroup[g] || []).sort();
      const block = document.createElement("div");
      block.innerHTML = `<h3>${esc(titles[g] || g)}</h3><div class="charts speedup-charts"></div>`;
      const speedEl = block.querySelector(".charts");
      ops.forEach(op => {
        const sp = Object.entries(dtypeColors).map(([dtype, color]) => ({
          color, dash: false,
          pts: (byOp[op] || []).filter(r => r.dtype===dtype && num(r.speedup) && r.openblas_status === "ok" && r.noblas_status === "ok")
            .sort((a,b)=>Number(a.n)-Number(b.n)).map(r=>({n:+r.n,y:+r.speedup,op:r.op})),
        }));
        speedEl.insertAdjacentHTML("beforeend", lineChart(op, "speedup", sp, {href: 1}));
      });
      speedHost.appendChild(block);
    });
    const tables = wrap.querySelector(".tables");
    Object.keys(byOp).sort().forEach(op => {
      const rs = byOp[op].sort((a,b) => a.dtype.localeCompare(b.dtype) || Number(a.n)-Number(b.n));
      const body = rs.map(r => {
        const oom = r.openblas_status !== "ok" || r.noblas_status !== "ok";
        const sp = num(r.speedup);
        return `<tr${oom ? " class='oom'" : ""}><td class='num'>${esc(r.n)}</td><td>${esc(r.dtype)}</td>
          <td class='num'>${esc(r.openblas_median_s)}</td><td class='num'>${esc(r.noblas_median_s)}</td>
          <td class='num'>${esc(r.openblas_gflops)}</td><td class='num'>${esc(r.noblas_gflops)}</td>
          <td class='num'>${sp ? sp.toFixed(3) : "—"}</td>
          <td>${esc(r.openblas_status)} / ${esc(r.noblas_status)}</td></tr>`;
      }).join("");
      tables.insertAdjacentHTML("beforeend", `<details>
        <summary>${esc(op)} <span class="caption">(${esc(rs[0].group)}${rs[0].blas ? " · " + esc(rs[0].blas) : ""})</span></summary>
        <table><thead><tr>
          <th class='num'>n</th><th>dtype</th>
          <th class='num'>${esc(left)} s</th><th class='num'>${esc(right)} s</th>
          <th class='num'>${esc(left)} GFLOPS</th><th class='num'>${esc(right)} GFLOPS</th>
          <th class='num'>speedup</th><th>status</th>
        </tr></thead><tbody>${body}</tbody></table></details>`);
    });
    host.appendChild(wrap);
  });
})();
</script>
</body>
</html>
"""
