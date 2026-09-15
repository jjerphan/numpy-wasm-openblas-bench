"""Compare OpenBLAS vs no-BLAS NumPy wasm CSVs and write report.html."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent


def geomean(xs: list[float]) -> float:
    xs = [x for x in xs if x > 0 and x == x]
    if not xs:
        return float("nan")
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def load_csv(path: Path) -> dict[tuple[str, str, str], dict]:
    out: dict[tuple[str, str, str], dict] = {}
    if not path.exists():
        return out
    with path.open() as f:
        for row in csv.DictReader(f):
            n = int(row.get("n") or 0)
            if n == 4096:
                continue
            if str(row.get("dtype") or "").startswith("complex"):
                continue
            if row.get("group") != "l1" and n > 1024:
                continue
            out[(row["op"], row["n"], row["dtype"])] = row
    return out


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


def load_meta(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text())


def compare_rows(openblas: dict, noblas: dict) -> list[dict]:
    keys = sorted(set(openblas) | set(noblas), key=lambda k: (k[0], int(k[1]), k[2]))
    rows = []
    for op, n, dtype in keys:
        a = openblas.get((op, n, dtype), {})
        b = noblas.get((op, n, dtype), {})
        ta = fnum(a.get("median_s"))
        tb = fnum(b.get("median_s"))
        ga = fnum(a.get("gflops"))
        gb = fnum(b.get("gflops"))
        speedup = ""
        if ta and tb and ta > 0:
            speedup = f"{tb / ta:.6f}"
        rows.append(
            {
                "op": op,
                "n": n,
                "dtype": dtype,
                "group": a.get("group") or b.get("group") or "",
                "blas": a.get("blas") or b.get("blas") or "",
                "openblas_median_s": a.get("median_s", ""),
                "noblas_median_s": b.get("median_s", ""),
                "openblas_gflops": a.get("gflops", ""),
                "noblas_gflops": b.get("gflops", ""),
                "speedup": speedup,
                "openblas_status": a.get("status", ""),
                "noblas_status": b.get("status", ""),
            }
        )
    return rows


def group_geomeans(rows: list[dict]) -> list[dict]:
    by: dict[str, list[float]] = {}
    all_sp: list[float] = []
    for row in rows:
        sp = fnum(row["speedup"])
        if sp is None:
            continue
        by.setdefault(row["group"] or "other", []).append(sp)
        all_sp.append(sp)
    order = ["l1", "l2", "l3", "lapack", "control"]
    out = []
    seen = set()
    for g in order:
        if g in by:
            out.append({"group": g, "n": len(by[g]), "geomean_speedup": geomean(by[g])})
            seen.add(g)
    for g, xs in sorted(by.items()):
        if g not in seen:
            out.append({"group": g, "n": len(xs), "geomean_speedup": geomean(xs)})
    out.append({"group": "ALL", "n": len(all_sp), "geomean_speedup": geomean(all_sp)})
    return out


def compare_and_report(
    openblas_csv: Path,
    noblas_csv: Path,
    openblas_meta: Path | None,
    noblas_meta: Path | None,
    compare_csv: Path,
    report_html: Path,
) -> None:
    from report import write_report

    ob = load_csv(openblas_csv)
    nb = load_csv(noblas_csv)
    rows = compare_rows(ob, nb)
    groups = group_geomeans(rows)

    fields = [
        "op",
        "n",
        "dtype",
        "group",
        "blas",
        "openblas_median_s",
        "noblas_median_s",
        "openblas_gflops",
        "noblas_gflops",
        "speedup",
        "openblas_status",
        "noblas_status",
    ]
    compare_csv.parent.mkdir(parents=True, exist_ok=True)
    with compare_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print(f"{'group':<16} {'n':>4} {'geomean_speedup':>16}")
    for g in groups:
        gm = g["geomean_speedup"]
        gm_s = f"{gm:.3f}" if gm == gm else "nan"
        print(f"{g['group']:<16} {g['n']:>4} {gm_s:>16}")

    write_report(
        report_html,
        rows=rows,
        groups=groups,
        openblas_meta=load_meta(openblas_meta),
        noblas_meta=load_meta(noblas_meta),
    )
    print(f"wrote {compare_csv}")
    print(f"wrote {report_html}")


def _cmp_payload(openblas_csv: Path, other_csv: Path, openblas_meta: Path | None, other_meta: Path | None) -> dict:
    from report import write_report as _  # noqa: F401

    ob = load_csv(openblas_csv)
    nb = load_csv(other_csv)
    rows = compare_rows(ob, nb)
    groups = group_geomeans(rows)
    om = load_meta(openblas_meta)
    nm = load_meta(other_meta)
    from report import LABEL_NAMES

    left_stem = str(om.get("label") or openblas_csv.stem)
    right_stem = str(nm.get("label") or other_csv.stem)
    return {
        "id": f"{left_stem}-vs-{right_stem}",
        "left_label": LABEL_NAMES.get(left_stem, left_stem),
        "right_label": LABEL_NAMES.get(right_stem, right_stem),
        "openblas": om,
        "noblas": nm,
        "groups": groups,
        "rows": rows,
    }


COMPARE_PAIRS = (
    ("ob034", "noblas"),
    ("ob034", "pyodide"),
    ("ob034", "cf64"),
    ("obdev", "ob034"),
    ("obdev", "pyodide"),
    ("obdev", "cf64"),
    ("obdev-relaxed", "obdev"),
    ("obdev-relaxed", "noblas"),
    ("obdev-relaxed", "cf64"),
)


def _labelled_csv(results: Path, stem: str) -> Path | None:
    path = results / f"{stem}.csv"
    return path if path.exists() else None


def _labelled_meta(results: Path, stem: str) -> Path | None:
    path = results / f"{stem}.meta.json"
    return path if path.exists() else None


def write_combined_index(results: Path) -> None:
    from report import write_report

    parts = []
    for left, right in COMPARE_PAIRS:
        left_csv = _labelled_csv(results, left)
        right_csv = _labelled_csv(results, right)
        if left_csv is None or right_csv is None:
            continue
        payload = _cmp_payload(
            left_csv,
            right_csv,
            _labelled_meta(results, left),
            _labelled_meta(results, right),
        )
        payload["id"] = f"{left}-vs-{right}"
        parts.append(payload)
    if not parts:
        print(f"no labelled CSV pairs in {results}; nothing to report")
        return
    write_report(
        results / "report.html",
        rows=parts[0]["rows"],
        groups=parts[0]["groups"],
        openblas_meta=parts[0]["openblas"],
        noblas_meta=parts[0]["noblas"],
        comparisons=parts,
    )
    print(f"wrote {results / 'report.html'}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("openblas", nargs="?", type=Path, default=HERE / "results" / "ob034.csv")
    p.add_argument("noblas", nargs="?", type=Path, default=HERE / "results" / "noblas.csv")
    p.add_argument("-o", "--output", type=Path, default=HERE / "results" / "compare.csv")
    p.add_argument("--report", type=Path, default=HERE / "results" / "report.html")
    args = p.parse_args()
    compare_and_report(
        args.openblas,
        args.noblas,
        args.openblas.with_suffix(".meta.json"),
        args.noblas.with_suffix(".meta.json"),
        args.output,
        args.report,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
