"""SVG figures for NUMPY_WASM_BLOG.md.

Two galleries, speedup only:
  s1_*  OpenBLAS 0.3.34 vs no-BLAS / Pyodide / conda-forge linux-64
  s2_*  OpenBLAS 0.3.35 vs 0.3.34 / Pyodide / conda-forge linux-64

Line colours are sampled from matplotlib's viridis colormap.
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
BLOG_DIR = HERE.parents[2] / "numpy-wasm-blog"

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


def fig_geomean_api(
    fast: list[dict],
    baselines: tuple[tuple[str, str], ...],
    prefix: str,
    title: str,
    xlabel: str,
) -> None:
    """Grouped horizontal bars: one cluster per API, one bar per baseline.

    Speedup is baseline time / featured time. Log x-axis so vs no-BLAS (~9×)
    and vs linux-64 (~0.3×) share a figure.
    """
    labels = [label for _op, label in API_GEOMEAN]
    n_api = len(labels)
    n_base = len(baselines)
    series = []
    for stem, name in baselines:
        slow = load_labeled(stem)
        series.append((name, [speedup_geomean(fast, slow, op) for op, _ in API_GEOMEAN]))
    finite = [v for _name, vals in series for v in vals if v == v]
    xmin = 0.04
    xmax = max(max(finite) * 1.55, 1.2)
    fig, ax = plt.subplots(figsize=(8.4, 7.2))
    y_centers = list(range(n_api - 1, -1, -1))
    group = 0.78
    height = group / n_base
    offsets = [(j - (n_base - 1) / 2) * height for j in range(n_base)]
    colors = viridis_samples(n_base, lo=0.12, hi=0.82)
    for (name, vals), offset, color in zip(series, offsets, colors):
        ys = [y + offset for y in y_centers]
        widths = [max(v - xmin, xmin * 0.01) if v == v else 0.0 for v in vals]
        bars = ax.barh(ys, widths, left=xmin, height=height * 0.9, color=color, label=name, zorder=2)
        for bar, v in zip(bars, vals):
            if v != v:
                continue
            ax.text(
                v * 1.04,
                bar.get_y() + bar.get_height() / 2,
                f"{v:.2f}×",
                ha="left",
                va="center",
                fontsize=8,
            )
    ax.axvline(1.0, color="k", linestyle="--", linewidth=1.0, zorder=1)
    ax.set_yticks(y_centers, labels)
    ax.set_xscale("log")
    ax.set_xlim(xmin, xmax)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=9)
    save(fig, f"{prefix}geomean_by_api.svg")


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
        for color, (stem, sname, dtype, style) in zip(colors, series):
            slow = load_labeled(stem)
            ns, ys = series_speedup(fast, slow, op, dtype)
            if not ns:
                continue
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
        ax.set_yscale("log")
        ax.set_xticks([64, 128, 256, 512, 1024])
        ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, _p: f"{int(x)}"))
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


def main() -> int:
    p = argparse.ArgumentParser()
    p.parse_args()
    style()
    for stem in ("noblas", "ob034", "obdev", "pyodide", "cf64"):
        path = RESULTS / f"{stem}.csv"
        if not path.exists():
            raise SystemExit(f"missing {path}")

    print(
        "s1 geomeans 0.3.34 vs no-BLAS",
        {
            l: round(speedup_geomean(load_labeled("ob034"), load_labeled("noblas"), o), 3)
            for o, l in API_GEOMEAN
        },
    )
    print(
        "s1 geomeans 0.3.34 vs linux-64",
        {
            l: round(speedup_geomean(load_labeled("ob034"), load_labeled("cf64"), o), 3)
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
        "s2 geomeans 0.3.35 vs linux-64",
        {
            l: round(speedup_geomean(load_labeled("obdev"), load_labeled("cf64"), o), 3)
            for o, l in API_GEOMEAN
        },
    )

    fig_geomean_api(
        load_labeled("ob034"),
        (
            ("noblas", "vs no BLAS"),
            ("pyodide", "vs Pyodide"),
            ("cf64", "vs linux-64"),
        ),
        "s1_",
        "OpenBLAS 0.3.34 vs no BLAS, Pyodide, and conda-forge linux-64",
        "geomean speedup (baseline time / 0.3.34 time)",
    )
    fig_speedup_vs_n(
        "ob034",
        (
            ("noblas", "vs no BLAS"),
            ("pyodide", "vs Pyodide"),
            ("cf64", "vs linux-64"),
        ),
        "s1_",
        "OpenBLAS 0.3.34 speedup versus n",
        "speedup (×)",
    )

    fig_geomean_api(
        load_labeled("obdev"),
        (
            ("ob034", "vs 0.3.34"),
            ("pyodide", "vs Pyodide"),
            ("cf64", "vs linux-64"),
        ),
        "s2_",
        "OpenBLAS 0.3.35 vs 0.3.34, Pyodide, and conda-forge linux-64",
        "geomean speedup (baseline time / 0.3.35 time)",
    )
    fig_speedup_vs_n(
        "obdev",
        (
            ("ob034", "vs 0.3.34"),
            ("pyodide", "vs Pyodide"),
            ("cf64", "vs linux-64"),
        ),
        "s2_",
        "OpenBLAS 0.3.35 speedup versus n",
        "speedup (×)",
    )

    BLOG_DIR.mkdir(parents=True, exist_ok=True)
    keep = {
        "s1_geomean_by_api.svg",
        "s1_speedup_vs_n.svg",
        "s2_geomean_by_api.svg",
        "s2_speedup_vs_n.svg",
    }
    for name in keep:
        shutil.copy2(OUT / name, BLOG_DIR / name)
        print(f"copied {BLOG_DIR / name}")
    for stale in BLOG_DIR.glob("*.svg"):
        if stale.name not in keep:
            stale.unlink()
            print(f"removed {stale}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
