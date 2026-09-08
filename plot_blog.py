"""SVG figures for NUMPY_WASM_BLOG.md.

Two galleries:
  s1_*  emscripten-forge no-BLAS vs OpenBLAS 0.3.34 (Pyodide as reference)
  s2_*  OpenBLAS 0.3.34 vs 0.3.35 (Pyodide as reference)

Primary y-axis is GFLOPS (or median seconds when GFLOPS is absent). Chart titles
name the OpenBLAS / emscripten-forge change, not Pyodide.
"""

from __future__ import annotations

import argparse
import csv
import math
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
    ("add", "np.add"),
)

LINALG_CURVES = (
    ("solve", "np.linalg.solve"),
    ("cholesky", "np.linalg.cholesky"),
    ("inv", "np.linalg.inv"),
    ("eigh", "np.linalg.eigh"),
)

# (csv stem, legend label, matplotlib color)
S1 = (
    ("ob034", "OpenBLAS 0.3.34", "#1f4e79"),
    ("noblas", "emscripten-forge (no BLAS)", "#a15c2d"),
    ("pyodide", "Pyodide", "#6b7280"),
)
S2 = (
    ("obdev", "OpenBLAS 0.3.35", "#1f7a4d"),
    ("ob034", "OpenBLAS 0.3.34", "#1f4e79"),
    ("pyodide", "Pyodide", "#6b7280"),
)


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


def series_gflops(rows: list[dict], op: str, dtype: str) -> tuple[list[int], list[float]]:
    pts = sorted(
        (
            (r["n"], r["gflops"])
            for r in rows
            if r["op"] == op and r["dtype"] == dtype and r["gflops"]
        ),
        key=lambda p: p[0],
    )
    if not pts:
        return [], []
    ns, ys = zip(*pts)
    return list(ns), list(ys)


def series_time(rows: list[dict], op: str, dtype: str) -> tuple[list[int], list[float]]:
    pts = sorted(
        (
            (r["n"], r["median_s"])
            for r in rows
            if r["op"] == op and r["dtype"] == dtype and r["median_s"]
        ),
        key=lambda p: p[0],
    )
    if not pts:
        return [], []
    ns, ys = zip(*pts)
    return list(ns), list(ys)


def style() -> None:
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
    slow: list[dict],
    prefix: str,
    title: str,
    xlabel: str,
    color: str = "#1f4e79",
) -> None:
    labels = [label for _op, label in API_GEOMEAN]
    vals = [speedup_geomean(fast, slow, op) for op, _label in API_GEOMEAN]
    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    y = range(len(labels) - 1, -1, -1)
    bars = ax.barh(list(y), vals, height=0.68, zorder=2, color=color)
    ax.axvline(1.0, color="k", linestyle="--", linewidth=1.0, zorder=1)
    ax.set_yticks(list(y), labels)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    finite = [v for v in vals if v == v]
    xmax = max(max(finite) * 1.22, 1.2)
    ax.set_xlim(0.0, xmax)
    span = max(finite)
    for bar, v in zip(bars, vals):
        if v != v:
            continue
        ax.text(
            v + span * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{v:.2f}×",
            ha="left",
            va="center",
            fontsize=10,
        )
    save(fig, f"{prefix}geomean_by_api.svg")


def _plot_lines(ax, suites, op: str, dtype: str, yfn) -> list[list[float]]:
    all_ys = []
    for stem, label, color in suites:
        rows = load_labeled(stem)
        ns, ys = yfn(rows, op, dtype)
        if not ns:
            continue
        ax.plot(ns, ys, marker="o", linewidth=1.8, markersize=6, label=label, color=color, zorder=3)
        all_ys.append(ys)
    return all_ys


def fig_matmul(suites, prefix: str, title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2), sharey=True)
    all_ys = []
    for ax, dtype in ((axes[0], "float32"), (axes[1], "float64")):
        ys = _plot_lines(ax, suites, "matmul", dtype, series_gflops)
        all_ys.extend(ys)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks([64, 128, 256, 512, 1024])
        ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, _p: f"{int(x)}"))
        ax.set_xlabel("n (square matrix)")
        ax.set_title(dtype)
    axes[0].set_ylabel("GFLOPS")
    axes[1].legend(loc="upper left", fontsize=9)
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    save(fig, f"{prefix}np_matmul.svg")


def fig_linalg(suites, prefix: str, title: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 7.0), sharex=True, sharey=True)
    for ax, (op, label) in zip(axes.flat, LINALG_CURVES):
        for stem, sname, color in suites:
            rows = load_labeled(stem)
            ns, ys = series_time(rows, op, "float64")
            if not ns:
                continue
            ls = {"ob034": "-", "obdev": "-", "noblas": "--", "pyodide": ":"}.get(stem, "-")
            ax.plot(
                ns,
                ys,
                marker="o",
                linewidth=1.8,
                markersize=5,
                linestyle=ls,
                color=color,
                label=sname if ax is axes[0, 0] else None,
                zorder=3,
            )
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks([64, 128, 256, 512, 1024])
        ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, _p: f"{int(x)}"))
        ax.set_title(label)
    axes[1, 0].set_xlabel("n (square matrix, float64)")
    axes[1, 1].set_xlabel("n (square matrix, float64)")
    axes[0, 0].set_ylabel("median time (s)")
    axes[1, 0].set_ylabel("median time (s)")
    axes[0, 0].legend(loc="upper left", fontsize=8)
    fig.suptitle(title, y=1.01)
    fig.tight_layout()
    save(fig, f"{prefix}np_linalg.svg")


def fig_matvec(suites, prefix: str, title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2), sharey=True)
    for ax, op, op_title in (
        (axes[0], "gemv_t", "x @ A"),
        (axes[1], "gemv_n", "A @ x"),
    ):
        for dtype, marker in (("float32", "o"), ("float64", "s")):
            for stem, sname, color in suites:
                rows = load_labeled(stem)
                ns, ys = series_gflops(rows, op, dtype)
                if not ns:
                    continue
                ls = "-" if dtype == "float32" else "--"
                ax.plot(
                    ns,
                    ys,
                    marker=marker,
                    linestyle=ls,
                    linewidth=1.8,
                    markersize=6,
                    color=color,
                    label=f"{sname} {dtype}",
                    zorder=3,
                    alpha=0.95 if stem != "pyodide" else 0.75,
                )
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks([64, 128, 256, 512, 1024])
        ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, _p: f"{int(x)}"))
        ax.set_xlabel("n")
        ax.set_title(op_title)
        ax.set_ylabel("GFLOPS")
        ax.legend(loc="upper left", fontsize=7)
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    save(fig, f"{prefix}np_matvec.svg")


def main() -> int:
    p = argparse.ArgumentParser()
    p.parse_args()
    style()
    for stem, *_ in (*S1, *S2):
        path = RESULTS / f"{stem}.csv"
        if not path.exists():
            raise SystemExit(f"missing {path}")

    print("s1 geomeans 0.3.34 vs no-BLAS", {l: round(speedup_geomean(load_labeled("ob034"), load_labeled("noblas"), o), 3) for o, l in API_GEOMEAN})
    print("s2 geomeans 0.3.35 vs 0.3.34", {l: round(speedup_geomean(load_labeled("obdev"), load_labeled("ob034"), o), 3) for o, l in API_GEOMEAN})

    fig_geomean_api(
        load_labeled("ob034"),
        load_labeled("noblas"),
        "s1_",
        "OpenBLAS 0.3.34 vs emscripten-forge NumPy without BLAS",
        "geomean speedup (no-BLAS time / 0.3.34 time)",
        color="#1f4e79",
    )
    fig_matmul(S1, "s1_", "np.matmul GFLOPS: OpenBLAS 0.3.34 vs NumPy without BLAS")
    fig_linalg(S1, "s1_", "np.linalg median time: OpenBLAS 0.3.34 vs NumPy without BLAS")
    fig_matvec(S1, "s1_", "A @ x / x @ A GFLOPS: OpenBLAS 0.3.34 vs NumPy without BLAS")

    fig_geomean_api(
        load_labeled("obdev"),
        load_labeled("ob034"),
        "s2_",
        "OpenBLAS 0.3.35 vs 0.3.34",
        "geomean speedup (0.3.34 time / 0.3.35 time)",
        color="#1f7a4d",
    )
    fig_matmul(S2, "s2_", "np.matmul GFLOPS: OpenBLAS 0.3.35 vs 0.3.34")
    fig_linalg(S2, "s2_", "np.linalg median time: OpenBLAS 0.3.35 vs 0.3.34")
    fig_matvec(S2, "s2_", "A @ x / x @ A GFLOPS: OpenBLAS 0.3.35 vs 0.3.34")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
