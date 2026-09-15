"""Run bench.py in two emscripten-forge wasm envs via pyjs-code-runner (Chromium)."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import socket
import sys
from contextlib import closing
from pathlib import Path

HERE = Path(__file__).resolve().parent
PIXI_ENVS = HERE / ".pixi" / "envs"
DEFAULT_OPENBLAS = PIXI_ENVS / "ob034"
DEFAULT_NOBLAS = PIXI_ENVS / "noblas"
RESULTS = HERE / "results"
WORK = HERE / "work"

L3_OPS = (
    "gemm",
    "gemm_f",
    "syrk",
    "batched_matmul",
    "gemm_dot",
    "matmul",
    "tensordot",
    "multi_dot",
    "matrix_power",
)
L3_DTYPES = ("float32", "float64")


def l3_skip_rows(n: int) -> list[list]:
    return [[op, n, dt] for op in L3_OPS for dt in L3_DTYPES]

# pyjs-code-runner hardcodes a 4-minute Playwright timeout; large GEMM exceeds that.
PLAYWRIGHT_TIMEOUT_MS = 0  # disable


def find_free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def load_done(path: Path) -> list[list]:
    if not path.exists():
        return []
    done = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line.startswith("{"):
                continue
            row = json.loads(line)
            if row.get("status"):
                done.append([row["op"], int(row["n"]), row["dtype"]])
    return done


def append_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def parse_console_text(text: str) -> tuple[str | None, object]:
    text = text.strip()
    # Playwright may prefix JSHandle / log levels.
    for prefix in ("META/", "ROW/"):
        idx = text.find(prefix)
        if idx >= 0:
            payload = text[idx + len(prefix) :]
            try:
                return prefix[:-1], json.loads(payload)
            except json.JSONDecodeError:
                return prefix[:-1], payload
    return None, None


async def run_playwright(
    page_url: str,
    work_dir: str,
    script: str,
    async_main: bool,
    js_utils: str,
    jsonl: Path | None = None,
):
    from playwright.async_api import async_playwright

    rows: list[dict] = []
    metas: list[dict] = []
    extra: list[str] = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        page.set_default_timeout(PLAYWRIGHT_TIMEOUT_MS)
        await page.goto(page_url)

        def handle_console(msg):
            txt = msg.text
            kind, payload = parse_console_text(txt)
            if kind == "ROW" and isinstance(payload, dict):
                rows.append(payload)
                if jsonl is not None:
                    append_jsonl(jsonl, [payload])
                print(
                    f"{payload.get('op'):<22} n={payload.get('n'):>6} "
                    f"{payload.get('dtype'):<8} {payload.get('status'):<12} "
                    f"{payload.get('median_s') or ''} {payload.get('gflops') or ''}",
                    flush=True,
                )
            elif kind == "META" and isinstance(payload, dict):
                metas.append(payload)
                print("META", json.dumps(payload), flush=True)
            else:
                extra.append(txt)
                if not txt.startswith("JSHandle") and "fetching" not in txt.lower():
                    print(txt, flush=True)

        page.on("console", handle_console)
        status = await page.evaluate(
            f"""async () => {{
                {js_utils}
                const print = (text) => {{ console.log(text) }}
                var pyjs = await make_pyjs(print, print, true);
                var r = globalThis.eval_main_script(pyjs, "{work_dir}", "{script}");
                if ({int(async_main)}) {{
                    r = r || await run_async_python_main(pyjs);
                }}
                return r;
            }}"""
        )
        version = browser.version
        await browser.close()
    return int(status or 0), rows, metas, extra, version


def pack_and_serve(conda_env: Path, mount_dir: Path, work_dir: str, script: str):
    from pyjs_code_runner.backend.backend import BackendType
    from pyjs_code_runner.backend.backend_base import BackendBase
    from pyjs_code_runner.constants import HTML_DIR, JS_DIR
    from pyjs_code_runner.run import copy_pyjs, pack_mounts
    from empack.pack import add_tarfile_to_env_meta, pack_env
    from empack.file_patterns import pkg_file_filter_from_yaml
    from empack.pack import DEFAULT_CONFIG_PATH as EMPACK_DEFAULT_CONFIG_PATH

    host_work_dir = WORK / conda_env.name
    if host_work_dir.exists():
        shutil.rmtree(host_work_dir)
    host_work_dir.mkdir(parents=True)

    copy_pyjs(
        conda_env=conda_env,
        backend_type=BackendType.browser_main,
        pyjs_dir=None,
        outdir=host_work_dir,
    )
    pkg_file_filter = pkg_file_filter_from_yaml(EMPACK_DEFAULT_CONFIG_PATH)
    cache_dir = HERE / ".empack-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    pack_env(
        env_prefix=conda_env,
        relocate_prefix="/",
        file_filters=pkg_file_filter,
        use_cache=True,
        cache_dir=cache_dir,
        outdir=host_work_dir,
        compresslevel=9,
    )
    env_meta_filename = host_work_dir / "empack_env_meta.json"
    mount_files = pack_mounts(
        mounts=[(mount_dir, Path(work_dir))],
        backend_type=BackendType.browser_main,
        host_work_dir=host_work_dir,
    )
    for mount_file in mount_files:
        tar = host_work_dir / mount_file
        try:
            add_tarfile_to_env_meta(env_meta_filename=env_meta_filename, tarfile=tar)
        except shutil.SameFileError:
            pass

    browser_main_html = "browser_main.html"
    shutil.copyfile(HTML_DIR / browser_main_html, host_work_dir / browser_main_html)
    shutil.copyfile(JS_DIR / "utils.js", host_work_dir / "utils.js")
    js_utils = BackendBase(
        host_work_dir=host_work_dir,
        work_dir=work_dir,
        script=script,
        async_main=False,
    ).js_utils()
    port = find_free_port()
    return host_work_dir, js_utils, port, browser_main_html


def overlay_runtime_openblas(conda_env: Path, meta: dict) -> None:
    """Replace NumPy show_config BLAS fields with the runtime OpenBLAS package.

    NumPy records scipy-openblas at build time; the conda lib is whatever
    openblas package is installed (version and TARGET from openblas.pc).
    """
    pc = conda_env / "lib" / "pkgconfig" / "openblas.pc"
    if not pc.exists():
        return
    version = None
    config = None
    for line in pc.read_text().splitlines():
        if line.startswith("version="):
            version = line.split("=", 1)[1].strip()
        elif line.startswith("openblas_config="):
            config = line.split("=", 1)[1].strip()
    meta_dir = conda_env / "conda-meta"
    pkg = None
    channel = None
    for p in sorted(meta_dir.glob("openblas-*.json")):
        rec = json.loads(p.read_text())
        pkg = f"{rec.get('version')}={rec.get('build')}"
        channel = rec.get("channel")
        if not version:
            version = rec.get("version")
        break
    meta["numpy_show_config_blas_name"] = meta.get("blas_name")
    meta["numpy_show_config_blas_version"] = meta.get("blas_version")
    meta["numpy_show_config_openblas_configuration"] = meta.get("blas_openblas_configuration")
    meta["blas_name"] = "openblas"
    if version:
        meta["blas_version"] = version
        meta["lapack_name"] = "openblas"
        meta["lapack_version"] = version
    if config:
        meta["blas_openblas_configuration"] = config
        meta["lapack_openblas_configuration"] = config
    if pkg:
        meta["openblas_conda"] = pkg
    if channel:
        meta["openblas_channel"] = channel


def run_env(
    label: str,
    conda_env: Path,
    expect_blas: bool,
    jsonl: Path,
    max_n: int | None,
    warmup: int,
    samples: int,
    resume: bool,
    extra_skip: list | None = None,
) -> dict:
    mount = HERE / "mount"
    if mount.exists():
        shutil.rmtree(mount)
    mount.mkdir()
    shutil.copyfile(HERE / "bench.py", mount / "bench.py")
    skip = load_done(jsonl) if resume else []
    if extra_skip:
        skip = list(skip) + list(extra_skip)
    (mount / "config.json").write_text(
        json.dumps(
            {
                "expect_blas": expect_blas,
                "max_n": max_n,
                "warmup": warmup,
                "samples": samples,
                "skip": skip,
            }
        )
    )

    work_dir = "/home/web_user"
    host_work_dir, js_utils, port, html = pack_and_serve(
        conda_env, mount, work_dir, "bench.py"
    )
    from pyjs_code_runner.backend.browser_main.server import server_context

    with server_context(work_dir=host_work_dir, port=port) as (_server, url):
        page_url = f"{url}/{html}"
        status, rows, metas, extra, chromium = asyncio.run(
            run_playwright(
                page_url, work_dir, "bench.py", False, js_utils, jsonl=jsonl
            )
        )
    meta = {}
    for m in metas:
        meta.update(m)
    meta["chromium"] = chromium
    meta["label"] = label
    meta["conda_env"] = str(conda_env)
    meta["return_code"] = status
    if expect_blas:
        overlay_runtime_openblas(conda_env, meta)
    (jsonl.with_suffix(".meta.json")).write_text(json.dumps(meta, indent=2, default=str))
    if status != 0 and not any(m.get("done") for m in metas):
        raise RuntimeError(f"{label} wasm run failed rc={status}")
    return meta


def jsonl_to_csv(jsonl: Path, csv_path: Path) -> None:
    import csv

    rows = []
    if jsonl.exists():
        with jsonl.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    rows = [
        r
        for r in rows
        if str(r.get("n")) != "4096"
        and not str(r.get("dtype") or "").startswith("complex")
        and (r.get("group") == "l1" or int(r.get("n") or 0) <= 1024)
    ]
    fields = [
        "op",
        "n",
        "dtype",
        "group",
        "blas",
        "inner",
        "median_s",
        "gflops",
        "min_s",
        "max_s",
        "status",
    ]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--openblas-env", type=Path, default=DEFAULT_OPENBLAS)
    p.add_argument("--noblas-env", type=Path, default=DEFAULT_NOBLAS)
    p.add_argument("--only", choices=("openblas", "noblas", "both"), default="both")
    p.add_argument(
        "--label",
        default=None,
        help="jsonl/csv stem when --only names a single env (default: ob034 or noblas)",
    )
    p.add_argument("--max-n", type=int, default=1024)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--samples", type=int, default=10)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--skip-run", action="store_true", help="only convert/compare existing results")
    p.add_argument(
        "--compare-pyodide",
        action="store_true",
        help="after converting, compare the labelled OpenBLAS CSV to results/pyodide.csv",
    )
    p.add_argument("--no-compare", action="store_true", help="skip OpenBLAS vs no-BLAS HTML")
    p.add_argument(
        "--skip-l3-n",
        type=int,
        action="append",
        default=[],
        metavar="N",
        help="skip Level-3 (GEMM-family) ops at size N (repeatable)",
    )
    args = p.parse_args()
    extra_skip = [row for n in args.skip_l3_n for row in l3_skip_rows(n)]

    RESULTS.mkdir(parents=True, exist_ok=True)
    metas = {}
    jobs = []
    if args.only in ("openblas", "both"):
        label = args.label if args.label and args.only == "openblas" else "ob034"
        jobs.append((label, args.openblas_env, True))
    if args.only in ("noblas", "both"):
        jobs.append(("noblas", args.noblas_env, False))

    if not args.skip_run:
        for label, env, expect in jobs:
            jsonl = RESULTS / f"{label}.jsonl"
            if not args.resume and jsonl.exists():
                jsonl.unlink()
            print(f"=== {label} {env} ===", flush=True)
            metas[label] = run_env(
                label,
                env,
                expect,
                jsonl,
                args.max_n,
                args.warmup,
                args.samples,
                args.resume,
                extra_skip=extra_skip,
            )

    # Always convert jsonls if present so --only / --resume cannot
    # pair a full run against a stale smoke-test CSV.
    labels = [j[0] for j in jobs] if jobs else ["ob034", "noblas"]
    if args.skip_run:
        extra = [args.label] if args.label else []
        labels = list(dict.fromkeys([*labels, *extra, "ob034", "noblas"]))
    for label in labels:
        jsonl = RESULTS / f"{label}.jsonl"
        if jsonl.exists():
            jsonl_to_csv(jsonl, RESULTS / f"{label}.csv")

    from compare import compare_and_report

    if args.compare_pyodide:
        stem = args.label or "ob034"
        left = RESULTS / f"{stem}.csv"
        if left.exists() and (RESULTS / "pyodide.csv").exists():
            compare_and_report(
                left,
                RESULTS / "pyodide.csv",
                RESULTS / f"{stem}.meta.json" if (RESULTS / f"{stem}.meta.json").exists() else None,
                RESULTS / "pyodide.meta.json" if (RESULTS / "pyodide.meta.json").exists() else None,
                RESULTS / f"compare_{stem}.csv",
                RESULTS / f"report_{stem}_pyodide.html",
            )

    if (
        not args.no_compare
        and (RESULTS / "ob034.csv").exists()
        and (RESULTS / "noblas.csv").exists()
    ):
        compare_and_report(
            RESULTS / "ob034.csv",
            RESULTS / "noblas.csv",
            RESULTS / "ob034.meta.json" if (RESULTS / "ob034.meta.json").exists() else None,
            RESULTS / "noblas.meta.json" if (RESULTS / "noblas.meta.json").exists() else None,
            RESULTS / "compare.csv",
            RESULTS / "report_noblas.html",
        )
        from compare import write_combined_index
        write_combined_index(RESULTS)
    return 0


if __name__ == "__main__":
    # Ensure host env packages are importable when invoked via micromamba run.
    os.chdir(HERE)
    raise SystemExit(main())
