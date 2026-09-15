"""SVG figures and appendix tables for the NumPy + OpenBLAS WASM blog post.

Galleries, speedup only:
  s1_*  OpenBLAS 0.3.34 vs no-BLAS (barplots and appendix curves)
  s2_*  OpenBLAS 0.3.35 (SIMD128) vs 0.3.34
  s3_*  0.3.34 / 0.3.35 SIMD128 / 0.3.35 Relaxed SIMD vs no-BLAS
  s4_*  OpenBLAS 0.3.35 Relaxed SIMD vs portable 0.3.35 (SIMD128)
  linux-64 stays in the generated tables, not on the graphs.

Line colours are sampled from matplotlib's viridis colormap.

Default output directory is results/blog/. Optional --copy-to DIR copies the
generated assets elsewhere (for example a blog asset folder).
"""

from __future__ import annotations

import argparse
import csv
import math
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("svg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
OUT = RESULTS / "blog"

API_GEOMEAN = (
    ("matmul", "np.matmul"),
    ("tensordot", "np.tensordot"),
    ("multi_dot", "np.linalg.multi_dot"),
    ("matrix_power", "np.linalg.matrix_power"),
    ("gemv_t", "x @ A"),
    ("solve", "np.linalg.solve"),
    ("inv", "np.linalg.inv"),
    ("cholesky", "np.linalg.cholesky"),
    ("qr", "np.linalg.qr"),
    ("eigh", "np.linalg.eigh"),
    ("lstsq", "np.linalg.lstsq"),
    ("svd", "np.linalg.svd"),
    ("gemv_n", "A @ x"),
    ("dot", "np.dot"),
)

# Speedup-versus-n: NumPy APIs that dominate typical linear-algebra workloads.
KEY_SPEEDUP_OPS = (
    ("matmul", "np.matmul"),
    ("solve", "np.linalg.solve"),
    ("cholesky", "np.linalg.cholesky"),
    ("inv", "np.linalg.inv"),
    ("eigh", "np.linalg.eigh"),
    ("gemv_t", "x @ A"),
    ("gemv_n", "A @ x"),
)


def viridis_samples(n: int, lo: float = 0.15, hi: float = 0.85) -> list:
    cmap = matplotlib.colormaps["viridis"]
    if n <= 1:
        return [cmap(0.55)]
    return [cmap(lo + (hi - lo) * i / (n - 1)) for i in range(n)]


def geomean(xs: list[float]) -> float:
    xs = [x for x in xs if x > 0 and x == x]
    return float(math.exp(sum(math.log(x) for x in xs) / len(xs))) if xs else float("nan")


def fnum(x) -> float | None:
    if x is None or x == "":
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if v != v:
        return None
    return v


def load_csv(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for row in csv.DictReader(f):
            if row.get("status") not in ("", "ok", None) and row.get("status") != "ok":
                continue
            if str(row.get("status") or "ok") != "ok":
                continue
            try:
                row["n"] = int(row["n"])
            except (TypeError, ValueError):
                continue
            row["gflops"] = fnum(row.get("gflops"))
            row["median_s"] = fnum(row.get("median_s"))
            rows.append(row)
    return rows


def load_labeled(stem: str) -> list[dict]:
    return load_csv(RESULTS / f"{stem}.csv")


def index_rows(rows: list[dict]) -> dict[tuple[str, int, str], dict]:
    return {(r["op"], r["n"], r["dtype"]): r for r in rows}


def speedup_geomean(fast: list[dict], slow: list[dict], op: str) -> float:
    a = index_rows(fast)
    b = index_rows(slow)
    xs = []
    for key, ra in a.items():
        if key[0] != op:
            continue
        rb = b.get(key)
        if not rb or not ra["median_s"] or not rb["median_s"] or ra["median_s"] <= 0:
            continue
        xs.append(rb["median_s"] / ra["median_s"])
    return geomean(xs)


def series_speedup(
    fast: list[dict],
    slow: list[dict],
    op: str,
    dtype: str,
) -> tuple[list[int], list[float]]:
    a = index_rows(fast)
    b = index_rows(slow)
    pts = []
    for (oop, n, dt), ra in a.items():
        if oop != op or dt != dtype:
            continue
        rb = b.get((op, n, dt))
        if not rb or not ra["median_s"] or not rb["median_s"] or ra["median_s"] <= 0:
            continue
        pts.append((n, rb["median_s"] / ra["median_s"]))
    pts.sort()
    if not pts:
        return [], []
    ns, ys = zip(*pts)
    return list(ns), list(ys)


def style() -> None:
    n = 6
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.28,
            "grid.linewidth": 0.7,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "svg.fonttype": "none",
            "legend.frameon": False,
            "axes.prop_cycle": plt.cycler(color=viridis_samples(n)),
            "image.cmap": "viridis",
        }
    )


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.savefig(path, format="svg", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"wrote {path}")


def _draw_geomean_bars(
    series: list[tuple[str, list[float]]],
    prefix: str,
    title: str,
    xlabel: str,
    parity_label: str = "Parity w.r.t. no BLAS",
) -> None:
    """Grouped horizontal bars shared by fig_geomean_api / fig_geomean_featured."""
    labels = [label for _op, label in API_GEOMEAN]
    n_api = len(labels)
    n_series = len(series)
    finite = [v for _name, vals in series for v in vals if v == v]
    xmax = max(max(finite) * 1.18, 1.2)
    # Wider / taller figure when several series need value labels + bottom legend.
    width = 8.4 if n_series <= 2 else 9.6
    fig, ax = plt.subplots(figsize=(width, 7.8))
    y_centers = list(range(n_api - 1, -1, -1))
    group = 0.78
    height = group / n_series
    offsets = [(j - (n_series - 1) / 2) * height for j in range(n_series)]
    colors = viridis_samples(n_series, lo=0.12, hi=0.82)
    for (name, vals), offset, color in zip(series, offsets, colors):
        ys = [y + offset for y in y_centers]
        widths = [v if v == v else 0.0 for v in vals]
        bars = ax.barh(ys, widths, height=height * 0.9, color=color, label=name, zorder=2)
        for bar, v in zip(bars, vals):
            if v != v:
                continue
            ax.text(
                v + xmax * 0.012,
                bar.get_y() + bar.get_height() / 2,
                f"{v:.2f}×",
                ha="left",
                va="center",
                fontsize=7 if n_series > 2 else 8,
            )
    parity_line = ax.axvline(
        1.0,
        color="k",
        linestyle=":",
        linewidth=1.2,
        zorder=1,
        label=parity_label,
    )
    ax.set_yticks(y_centers, labels)
    ax.set_xlim(0, xmax)
    ax.set_xlabel(xlabel, labelpad=10)
    ax.set_title(title, pad=12)
    handles, leg_labels = ax.get_legend_handles_labels()
    # Single-series plots: only the parity reference (title already names the stack).
    # Multi-series: stack labels first, parity last.
    if n_series == 1:
        handles, leg_labels = [parity_line], [parity_label]
    else:
        bar_handles = [h for h, lab in zip(handles, leg_labels) if lab != parity_label]
        bar_labels = [lab for lab in leg_labels if lab != parity_label]
        handles = bar_handles + [parity_line]
        leg_labels = bar_labels + [parity_label]
    # Legend below the x-label (axes coordinates); leave room so they do not collide.
    ax.legend(
        handles,
        leg_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=min(len(handles), 4),
        fontsize=9,
        frameon=False,
        borderaxespad=0.0,
    )
    fig.subplots_adjust(bottom=0.20 if n_series == 1 else 0.24)
    save(fig, f"{prefix}geomean_by_api.svg")


def fig_geomean_api(
    fast: list[dict],
    baselines: tuple[tuple[str, str], ...],
    prefix: str,
    title: str,
    xlabel: str,
    parity_label: str = "Parity w.r.t. no BLAS",
) -> None:
    """Grouped horizontal bars: one cluster per API, one bar per baseline.

    Speedup is baseline time / featured time. Linear x-axis so bar length
    is proportional to the speedup (1× is the dotted reference).
    """
    series = []
    for stem, name in baselines:
        slow = load_labeled(stem)
        series.append((name, [speedup_geomean(fast, slow, op) for op, _ in API_GEOMEAN]))
    _draw_geomean_bars(series, prefix, title, xlabel, parity_label=parity_label)


def fig_geomean_featured(
    featured: tuple[tuple[str, str], ...],
    baseline_stem: str,
    prefix: str,
    title: str,
    xlabel: str,
    parity_label: str = "Parity w.r.t. no BLAS",
) -> None:
    """Grouped horizontal bars: one cluster per API, one bar per featured stack.

    Speedup is baseline time / featured time (same baseline for every series).
    """
    slow = load_labeled(baseline_stem)
    series = []
    for stem, name in featured:
        fast = load_labeled(stem)
        series.append((name, [speedup_geomean(fast, slow, op) for op, _ in API_GEOMEAN]))
    _draw_geomean_bars(series, prefix, title, xlabel, parity_label=parity_label)


def fig_speedup_vs_n(
    fast_stem: str,
    baselines: tuple[tuple[str, str], ...],
    prefix: str,
    title: str,
    ylabel: str,
) -> None:
    """Small multiples: speedup vs n for KEY_SPEEDUP_OPS.

    Each baseline is (csv_stem, legend label). float32 is solid with points;
    float64 is dashed with ×. Colours are viridis samples, one per
    (baseline, dtype) series.
    """
    fast = load_labeled(fast_stem)
    # Matplotlib names: "." is point, "x" is ×. Point is drawn smaller than
    # circle/"o", so give it a larger markersize; thicken × so it stays visible.
    dtype_style = {
        "float32": dict(marker=".", linestyle="-", markersize=10, markeredgewidth=1.0),
        "float64": dict(marker="x", linestyle="--", markersize=7, markeredgewidth=1.4),
    }
    series = []
    for stem, sname in baselines:
        for dtype in ("float32", "float64"):
            series.append((stem, sname, dtype, dtype_style[dtype]))
    colors = viridis_samples(len(series))
    n_ops = len(KEY_SPEEDUP_OPS)
    ncols = 2
    nrows = (n_ops + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(9.2, 3.15 * nrows), sharex=True)
    axes_flat = list(axes.flat)
    for i, (op, label) in enumerate(KEY_SPEEDUP_OPS):
        ax = axes_flat[i]
        plotted_ys: list[float] = []
        for color, (stem, sname, dtype, style) in zip(colors, series):
            slow = load_labeled(stem)
            ns, ys = series_speedup(fast, slow, op, dtype)
            if not ns:
                continue
            plotted_ys.extend(ys)
            ax.plot(
                ns,
                ys,
                linewidth=1.7,
                color=color,
                label=f"{sname} {dtype}",
                zorder=3,
                **style,
            )
        ax.axhline(1.0, color="k", linestyle=":", linewidth=0.9, zorder=1)
        ax.set_xscale("log", base=2)
        ax.set_xticks([64, 128, 256, 512, 1024])
        ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, _p: f"{int(x)}"))
        # Start at 1× unless some speedups fall below 1: then start slightly
        # below the observed minimum (not at 0).
        if plotted_ys and any(y < 1.0 for y in plotted_ys):
            ymin = min(plotted_ys)
            # Small pad below the floor; keep a positive lower bound.
            ax.set_ylim(bottom=max(0.05, ymin * 0.92))
        else:
            ax.set_ylim(bottom=1.0)
        ax.set_title(label)
        if i % ncols == 0:
            ax.set_ylabel(ylabel)
    unused = axes_flat[n_ops:]
    for ax in unused:
        ax.axis("off")
    if unused:
        handles, labels = axes_flat[0].get_legend_handles_labels()
        unused[0].legend(handles, labels, loc="center", fontsize=8, frameon=False)
    for ax in axes_flat[max(0, n_ops - ncols) : n_ops]:
        ax.set_xlabel("n")
    fig.suptitle(title, y=1.01)
    fig.tight_layout()
    save(fig, f"{prefix}speedup_vs_n.svg")


def _fmt_gflops(x: float | None) -> str:
    if x is None or x != x:
        return "—"
    if x >= 100:
        return f"{x:.1f}"
    if x >= 10:
        return f"{x:.1f}"
    return f"{x:.2f}"


def _fmt_ms(seconds: float | None) -> str:
    if seconds is None or seconds != seconds:
        return "—"
    return f"{seconds * 1000:.0f}"


def _fmt_speedup(x: float | None) -> str:
    if x is None or x != x:
        return "—"
    return f"{x:.2f}×"


def _lookup(ix: dict, op: str, n: int, dtype: str) -> dict | None:
    return ix.get((op, n, dtype))


def _speedup(fast: dict | None, slow: dict | None) -> float | None:
    if not fast or not slow:
        return None
    a, b = fast.get("median_s"), slow.get("median_s")
    if not a or not b or a <= 0:
        return None
    return b / a


def write_appendix_tables(have_cf64: bool, have_relaxed: bool) -> None:
    """Markdown tables embedded in NUMPY_WASM_BLOG.md appendices."""
    noblas = index_rows(load_labeled("noblas"))
    ob034 = index_rows(load_labeled("ob034"))
    obdev = index_rows(load_labeled("obdev"))
    relaxed = index_rows(load_labeled("obdev-relaxed")) if have_relaxed else {}
    cf64 = index_rows(load_labeled("cf64")) if have_cf64 else {}
    noblas_rows = load_labeled("noblas")
    ob034_rows = load_labeled("ob034")
    obdev_rows = load_labeled("obdev")
    relaxed_rows = load_labeled("obdev-relaxed") if have_relaxed else []
    cf64_rows = load_labeled("cf64") if have_cf64 else []

    lines: list[str] = ["<!-- Generated by plot_blog.py; do not edit by hand. -->", ""]
    lines.append("### Geometric-mean speedups by API")
    lines.append("")
    hdr = "| API | 0.3.34 vs no BLAS"
    sep = "| --- | ---:"
    if have_cf64:
        hdr += " | 0.3.34 vs linux-64"
        sep += " | ---:"
    hdr += " | 0.3.35 vs 0.3.34 | 0.3.35 vs no BLAS"
    sep += " | ---: | ---:"
    if have_cf64:
        hdr += " | 0.3.35 vs linux-64"
        sep += " | ---:"
    if have_relaxed:
        hdr += " | Relaxed SIMD vs 0.3.35 | Relaxed SIMD vs no BLAS"
        sep += " | ---: | ---:"
        if have_cf64:
            hdr += " | Relaxed SIMD vs linux-64"
            sep += " | ---:"
    hdr += " |"
    sep += " |"
    lines.extend([hdr, sep])
    for op, label in API_GEOMEAN:
        row = [f"| `{label}`"]
        row.append(_fmt_speedup(speedup_geomean(ob034_rows, noblas_rows, op)))
        if have_cf64:
            row.append(_fmt_speedup(speedup_geomean(ob034_rows, cf64_rows, op)))
        row.append(_fmt_speedup(speedup_geomean(obdev_rows, ob034_rows, op)))
        row.append(_fmt_speedup(speedup_geomean(obdev_rows, noblas_rows, op)))
        if have_cf64:
            row.append(_fmt_speedup(speedup_geomean(obdev_rows, cf64_rows, op)))
        if have_relaxed:
            row.append(_fmt_speedup(speedup_geomean(relaxed_rows, obdev_rows, op)))
            row.append(_fmt_speedup(speedup_geomean(relaxed_rows, noblas_rows, op)))
            if have_cf64:
                row.append(_fmt_speedup(speedup_geomean(relaxed_rows, cf64_rows, op)))
        lines.append(" | ".join(row) + " |")
    lines.append("")

    lines.append("### `np.matmul` across sizes (GFLOPS)")
    lines.append("")
    h = "| n | dtype | no BLAS | OpenBLAS 0.3.34 | OpenBLAS 0.3.35"
    s = "| ---: | --- | ---: | ---: | ---:"
    if have_relaxed:
        h += " | 0.3.35 Relaxed SIMD"
        s += " | ---:"
    if have_cf64:
        h += " | linux-64"
        s += " | ---:"
    h += " | 0.3.34 vs no BLAS | 0.3.35 vs 0.3.34 | 0.3.35 vs no BLAS"
    s += " | ---: | ---: | ---:"
    if have_relaxed:
        h += " | Relaxed SIMD vs 0.3.35 | Relaxed SIMD vs no BLAS"
        s += " | ---: | ---:"
    if have_cf64:
        h += " | 0.3.34 vs linux-64 | 0.3.35 vs linux-64"
        s += " | ---: | ---:"
        if have_relaxed:
            h += " | Relaxed SIMD vs linux-64"
            s += " | ---:"
    h += " |"
    s += " |"
    lines.extend([h, s])
    for n in (64, 128, 256, 512, 1024):
        for dtype in ("float32", "float64"):
            a = _lookup(noblas, "matmul", n, dtype)
            b = _lookup(ob034, "matmul", n, dtype)
            c = _lookup(obdev, "matmul", n, dtype)
            r = _lookup(relaxed, "matmul", n, dtype) if have_relaxed else None
            d = _lookup(cf64, "matmul", n, dtype) if have_cf64 else None
            cells = [
                f"| {n}",
                f"`{dtype}`",
                _fmt_gflops(a["gflops"] if a else None),
                f"**{_fmt_gflops(b['gflops'] if b else None)}**",
                f"**{_fmt_gflops(c['gflops'] if c else None)}**",
            ]
            if have_relaxed:
                cells.append(f"**{_fmt_gflops(r['gflops'] if r else None)}**")
            if have_cf64:
                cells.append(_fmt_gflops(d["gflops"] if d else None))
            cells.append(_fmt_speedup(_speedup(b, a)))
            cells.append(_fmt_speedup(_speedup(c, b)))
            cells.append(_fmt_speedup(_speedup(c, a)))
            if have_relaxed:
                cells.append(_fmt_speedup(_speedup(r, c)))
                cells.append(_fmt_speedup(_speedup(r, a)))
            if have_cf64:
                cells.append(_fmt_speedup(_speedup(b, d)))
                cells.append(_fmt_speedup(_speedup(c, d)))
                if have_relaxed:
                    cells.append(_fmt_speedup(_speedup(r, d)))
            lines.append(" | ".join(cells) + " |")
    lines.append("")

    lines.append("### `np.linalg` at `n = 1024` (median time, ms)")
    lines.append("")
    h = "| call | dtype | no BLAS | OpenBLAS 0.3.34 | OpenBLAS 0.3.35"
    s = "| --- | --- | ---: | ---: | ---:"
    if have_relaxed:
        h += " | 0.3.35 Relaxed SIMD"
        s += " | ---:"
    if have_cf64:
        h += " | linux-64"
        s += " | ---:"
    h += " | 0.3.34 vs no BLAS | 0.3.35 vs 0.3.34 | 0.3.35 vs no BLAS"
    s += " | ---: | ---: | ---:"
    if have_relaxed:
        h += " | Relaxed SIMD vs 0.3.35 | Relaxed SIMD vs no BLAS"
        s += " | ---: | ---:"
    if have_cf64:
        h += " | 0.3.34 vs linux-64 | 0.3.35 vs linux-64"
        s += " | ---: | ---:"
        if have_relaxed:
            h += " | Relaxed SIMD vs linux-64"
            s += " | ---:"
    h += " |"
    s += " |"
    lines.extend([h, s])
    linalg_ops = (
        ("solve", "np.linalg.solve"),
        ("cholesky", "np.linalg.cholesky"),
        ("qr", "np.linalg.qr"),
        ("inv", "np.linalg.inv"),
        ("eigh", "np.linalg.eigh"),
        ("svd", "np.linalg.svd"),
    )
    for op, label in linalg_ops:
        for dtype in ("float32", "float64"):
            a = _lookup(noblas, op, 1024, dtype)
            b = _lookup(ob034, op, 1024, dtype)
            c = _lookup(obdev, op, 1024, dtype)
            r = _lookup(relaxed, op, 1024, dtype) if have_relaxed else None
            d = _lookup(cf64, op, 1024, dtype) if have_cf64 else None
            cells = [
                f"| `{label}`",
                f"`{dtype}`",
                _fmt_ms(a["median_s"] if a else None),
                f"**{_fmt_ms(b['median_s'] if b else None)}**",
                f"**{_fmt_ms(c['median_s'] if c else None)}**",
            ]
            if have_relaxed:
                cells.append(f"**{_fmt_ms(r['median_s'] if r else None)}**")
            if have_cf64:
                cells.append(_fmt_ms(d["median_s"] if d else None))
            cells.append(_fmt_speedup(_speedup(b, a)))
            cells.append(_fmt_speedup(_speedup(c, b)))
            cells.append(_fmt_speedup(_speedup(c, a)))
            if have_relaxed:
                cells.append(_fmt_speedup(_speedup(r, c)))
                cells.append(_fmt_speedup(_speedup(r, a)))
            if have_cf64:
                cells.append(_fmt_speedup(_speedup(b, d)))
                cells.append(_fmt_speedup(_speedup(c, d)))
                if have_relaxed:
                    cells.append(_fmt_speedup(_speedup(r, d)))
            lines.append(" | ".join(cells) + " |")
    lines.append("")

    lines.append("### Matrix–vector `@` across sizes (GFLOPS)")
    lines.append("")
    h = "| call | n | dtype | no BLAS | OpenBLAS 0.3.34 | OpenBLAS 0.3.35"
    s = "| --- | ---: | --- | ---: | ---: | ---:"
    if have_relaxed:
        h += " | 0.3.35 Relaxed SIMD"
        s += " | ---:"
    if have_cf64:
        h += " | linux-64"
        s += " | ---:"
    h += " | 0.3.34 vs no BLAS | 0.3.35 vs 0.3.34 | 0.3.35 vs no BLAS"
    s += " | ---: | ---: | ---:"
    if have_relaxed:
        h += " | Relaxed SIMD vs 0.3.35 | Relaxed SIMD vs no BLAS"
        s += " | ---: | ---:"
    if have_cf64:
        h += " | 0.3.34 vs linux-64 | 0.3.35 vs linux-64"
        s += " | ---: | ---:"
        if have_relaxed:
            h += " | Relaxed SIMD vs linux-64"
            s += " | ---:"
    h += " |"
    s += " |"
    lines.extend([h, s])
    for op, label in (("gemv_t", "x @ A"), ("gemv_n", "A @ x")):
        for n in (64, 128, 256, 512, 1024):
            for dtype in ("float32", "float64"):
                a = _lookup(noblas, op, n, dtype)
                b = _lookup(ob034, op, n, dtype)
                c = _lookup(obdev, op, n, dtype)
                r = _lookup(relaxed, op, n, dtype) if have_relaxed else None
                d = _lookup(cf64, op, n, dtype) if have_cf64 else None
                cells = [
                    f"| `{label}`",
                    str(n),
                    f"`{dtype}`",
                    _fmt_gflops(a["gflops"] if a else None),
                    _fmt_gflops(b["gflops"] if b else None),
                    f"**{_fmt_gflops(c['gflops'] if c else None)}**",
                ]
                if have_relaxed:
                    cells.append(f"**{_fmt_gflops(r['gflops'] if r else None)}**")
                if have_cf64:
                    cells.append(_fmt_gflops(d["gflops"] if d else None))
                cells.append(_fmt_speedup(_speedup(b, a)))
                cells.append(_fmt_speedup(_speedup(c, b)))
                cells.append(_fmt_speedup(_speedup(c, a)))
                if have_relaxed:
                    cells.append(_fmt_speedup(_speedup(r, c)))
                    cells.append(_fmt_speedup(_speedup(r, a)))
                if have_cf64:
                    cells.append(_fmt_speedup(_speedup(b, d)))
                    cells.append(_fmt_speedup(_speedup(c, d)))
                    if have_relaxed:
                        cells.append(_fmt_speedup(_speedup(r, d)))
                lines.append(" | ".join(cells) + " |")
    lines.append("")

    # Dedicated appendix: 0.3.35 (SIMD128) versus no BLAS.
    lines.append("<!-- APPENDIX_035_VS_NOBLAS -->")
    lines.append("### `np.matmul` across sizes (GFLOPS)")
    lines.append("")
    lines.extend(
        [
            "| n | dtype | no BLAS | OpenBLAS 0.3.35 | 0.3.35 vs no BLAS |",
            "| ---: | --- | ---: | ---: | ---: |",
        ]
    )
    for n in (64, 128, 256, 512, 1024):
        for dtype in ("float32", "float64"):
            a = _lookup(noblas, "matmul", n, dtype)
            c = _lookup(obdev, "matmul", n, dtype)
            lines.append(
                " | ".join(
                    [
                        f"| {n}",
                        f"`{dtype}`",
                        _fmt_gflops(a["gflops"] if a else None),
                        f"**{_fmt_gflops(c['gflops'] if c else None)}**",
                        _fmt_speedup(_speedup(c, a)),
                    ]
                )
                + " |"
            )
    lines.append("")
    lines.append("### Matrix–vector `@` across sizes (GFLOPS)")
    lines.append("")
    lines.extend(
        [
            "| call | n | dtype | no BLAS | OpenBLAS 0.3.35 | 0.3.35 vs no BLAS |",
            "| --- | ---: | --- | ---: | ---: | ---: |",
        ]
    )
    for op, label in (("gemv_t", "x @ A"), ("gemv_n", "A @ x")):
        for n in (64, 128, 256, 512, 1024):
            for dtype in ("float32", "float64"):
                a = _lookup(noblas, op, n, dtype)
                c = _lookup(obdev, op, n, dtype)
                lines.append(
                    " | ".join(
                        [
                            f"| `{label}`",
                            str(n),
                            f"`{dtype}`",
                            _fmt_gflops(a["gflops"] if a else None),
                            f"**{_fmt_gflops(c['gflops'] if c else None)}**",
                            _fmt_speedup(_speedup(c, a)),
                        ]
                    )
                    + " |"
                )
    lines.append("")

    if have_relaxed:
        lines.append("<!-- APPENDIX_RELAXED_VS_035 -->")
        lines.append("### Geometric-mean speedups by API (Relaxed SIMD vs SIMD128)")
        lines.append("")
        lines.extend(
            [
                "| API | Relaxed SIMD vs 0.3.35 | Relaxed SIMD vs no BLAS |",
                "| --- | ---: | ---: |",
            ]
        )
        for op, label in API_GEOMEAN:
            lines.append(
                " | ".join(
                    [
                        f"| `{label}`",
                        _fmt_speedup(speedup_geomean(relaxed_rows, obdev_rows, op)),
                        _fmt_speedup(speedup_geomean(relaxed_rows, noblas_rows, op)),
                    ]
                )
                + " |"
            )
        lines.append("")
        lines.append("### `np.matmul` across sizes (GFLOPS)")
        lines.append("")
        lines.extend(
            [
                "| n | dtype | OpenBLAS 0.3.35 | 0.3.35 Relaxed SIMD | Relaxed SIMD vs 0.3.35 | Relaxed SIMD vs no BLAS |",
                "| ---: | --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for n in (64, 128, 256, 512, 1024):
            for dtype in ("float32", "float64"):
                a = _lookup(noblas, "matmul", n, dtype)
                c = _lookup(obdev, "matmul", n, dtype)
                r = _lookup(relaxed, "matmul", n, dtype)
                lines.append(
                    " | ".join(
                        [
                            f"| {n}",
                            f"`{dtype}`",
                            _fmt_gflops(c["gflops"] if c else None),
                            f"**{_fmt_gflops(r['gflops'] if r else None)}**",
                            _fmt_speedup(_speedup(r, c)),
                            _fmt_speedup(_speedup(r, a)),
                        ]
                    )
                    + " |"
                )
        lines.append("")
        lines.append("### Matrix–vector `@` across sizes (GFLOPS)")
        lines.append("")
        lines.extend(
            [
                "| call | n | dtype | OpenBLAS 0.3.35 | 0.3.35 Relaxed SIMD | Relaxed SIMD vs 0.3.35 | Relaxed SIMD vs no BLAS |",
                "| --- | ---: | --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for op, label in (("gemv_t", "x @ A"), ("gemv_n", "A @ x")):
            for n in (64, 128, 256, 512, 1024):
                for dtype in ("float32", "float64"):
                    a = _lookup(noblas, op, n, dtype)
                    c = _lookup(obdev, op, n, dtype)
                    r = _lookup(relaxed, op, n, dtype)
                    lines.append(
                        " | ".join(
                            [
                                f"| `{label}`",
                                str(n),
                                f"`{dtype}`",
                                _fmt_gflops(c["gflops"] if c else None),
                                f"**{_fmt_gflops(r['gflops'] if r else None)}**",
                                _fmt_speedup(_speedup(r, c)),
                                _fmt_speedup(_speedup(r, a)),
                            ]
                        )
                        + " |"
                    )
        lines.append("")

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "appendix_tables.md"
    path.write_text("\n".join(lines) + "\n")
    print(f"wrote {path}")


def main() -> int:
    global OUT
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out",
        type=Path,
        default=RESULTS / "blog",
        help="directory for SVG figures and appendix_tables.md (default: results/blog)",
    )
    p.add_argument(
        "--copy-to",
        type=Path,
        default=None,
        help="optional extra directory to copy generated assets into",
    )
    args = p.parse_args()
    OUT = args.out
    OUT.mkdir(parents=True, exist_ok=True)
    style()
    required = ("noblas", "ob034", "obdev")
    optional = ("cf64", "obdev-relaxed")
    for stem in required:
        path = RESULTS / f"{stem}.csv"
        if not path.exists():
            raise SystemExit(f"missing {path}")
    have = {stem: (RESULTS / f"{stem}.csv").exists() for stem in optional}

    print(
        "s1 geomeans 0.3.34 vs no-BLAS",
        {
            l: round(speedup_geomean(load_labeled("ob034"), load_labeled("noblas"), o), 3)
            for o, l in API_GEOMEAN
        },
    )
    print(
        "s2 geomeans 0.3.35 vs 0.3.34",
        {
            l: round(speedup_geomean(load_labeled("obdev"), load_labeled("ob034"), o), 3)
            for o, l in API_GEOMEAN
        },
    )
    print(
        "s3 geomeans 0.3.35 vs no-BLAS",
        {
            l: round(speedup_geomean(load_labeled("obdev"), load_labeled("noblas"), o), 3)
            for o, l in API_GEOMEAN
        },
    )
    if have["obdev-relaxed"]:
        print(
            "s4 geomeans relaxed vs 0.3.35",
            {
                l: round(
                    speedup_geomean(load_labeled("obdev-relaxed"), load_labeled("obdev"), o),
                    3,
                )
                for o, l in API_GEOMEAN
            },
        )
    if have["cf64"]:
        print(
            "s1 geomeans 0.3.34 vs linux-64",
            {
                l: round(speedup_geomean(load_labeled("ob034"), load_labeled("cf64"), o), 3)
                for o, l in API_GEOMEAN
            },
        )
        print(
            "s2 geomeans 0.3.35 vs linux-64",
            {
                l: round(speedup_geomean(load_labeled("obdev"), load_labeled("cf64"), o), 3)
                for o, l in API_GEOMEAN
            },
        )

    # Graphs: only the primary WASM comparison (no Pyodide, no linux-64).
    # linux-64 remains in the appendix tables.
    s1_bar = (("noblas", "vs no BLAS"),)
    s2_bar = (("ob034", "vs 0.3.34"),)
    s1_app: list[tuple[str, str]] = [("noblas", "vs no BLAS")]
    s2_app: list[tuple[str, str]] = [("ob034", "vs 0.3.34")]

    fig_geomean_api(
        load_labeled("ob034"),
        s1_bar,
        "s1_",
        "OpenBLAS 0.3.34 vs no BLAS Geometric-mean speedup by API",
        "geomean speedup (no-BLAS time / 0.3.34 time)",
        parity_label="Parity w.r.t. no BLAS",
    )
    fig_speedup_vs_n(
        "ob034",
        tuple(s1_app),
        "app_s1_",
        "OpenBLAS 0.3.34 vs no BLAS w.r.t input size (n)",
        "speedup (×)",
    )

    fig_geomean_api(
        load_labeled("obdev"),
        s2_bar,
        "s2_",
        "OpenBLAS 0.3.35 vs 0.3.34 Geometric-mean speedup by API",
        "geomean speedup (0.3.34 time / 0.3.35 time)",
        parity_label="Parity w.r.t. OpenBLAS 0.3.34",
    )
    fig_speedup_vs_n(
        "obdev",
        tuple(s2_app),
        "app_s2_",
        "OpenBLAS 0.3.35 vs 0.3.34 w.r.t input size (n)",
        "speedup (×)",
    )

    s3_featured = (
        ("ob034", "OpenBLAS 0.3.34"),
        ("obdev", "OpenBLAS 0.3.35 (SIMD128)"),
    )
    if have["obdev-relaxed"]:
        s3_featured = s3_featured + (("obdev-relaxed", "OpenBLAS 0.3.35 (Relaxed SIMD)"),)
    fig_geomean_featured(
        s3_featured,
        "noblas",
        "s3_",
        "OpenBLAS vs no BLAS Geometric-mean speedup by API",
        "geomean speedup (no-BLAS time / OpenBLAS time)",
        parity_label="Parity w.r.t. no BLAS",
    )
    fig_speedup_vs_n(
        "obdev-relaxed" if have["obdev-relaxed"] else "obdev",
        (("noblas", "vs no BLAS"),),
        "app_s3_",
        (
            "OpenBLAS 0.3.35 Relaxed SIMD vs no BLAS w.r.t input size (n)"
            if have["obdev-relaxed"]
            else "OpenBLAS 0.3.35 vs no BLAS w.r.t input size (n)"
        ),
        "speedup (×)",
    )

    if have["obdev-relaxed"]:
        fig_geomean_api(
            load_labeled("obdev-relaxed"),
            (("obdev", "vs 0.3.35 SIMD128"),),
            "s4_",
            "OpenBLAS 0.3.35 Relaxed SIMD vs SIMD128 Geometric-mean speedup by API",
            "geomean speedup (SIMD128 time / Relaxed SIMD time)",
            parity_label="Parity w.r.t. OpenBLAS 0.3.35 (SIMD128)",
        )
        fig_speedup_vs_n(
            "obdev-relaxed",
            (("obdev", "vs 0.3.35 SIMD128"),),
            "app_s4_",
            "OpenBLAS 0.3.35 Relaxed SIMD vs SIMD128 w.r.t input size (n)",
            "speedup (×)",
        )

    write_appendix_tables(have["cf64"], have["obdev-relaxed"])

    keep = {
        "s1_geomean_by_api.svg",
        "s2_geomean_by_api.svg",
        "s3_geomean_by_api.svg",
        "app_s1_speedup_vs_n.svg",
        "app_s2_speedup_vs_n.svg",
        "app_s3_speedup_vs_n.svg",
        "appendix_tables.md",
    }
    if have["obdev-relaxed"]:
        keep |= {"s4_geomean_by_api.svg", "app_s4_speedup_vs_n.svg"}
    print(f"wrote figures under {OUT}")
    if args.copy_to is not None:
        dest = args.copy_to
        dest.mkdir(parents=True, exist_ok=True)
        for name in keep:
            src = OUT / name
            if not src.exists():
                continue
            shutil.copy2(src, dest / name)
            print(f"copied {dest / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
