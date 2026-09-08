"""Run bench.py in official Pyodide numpy (CDN) via headless Chromium."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import socket
import sys
import threading
from contextlib import closing
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PYODIDE_INDEX = "https://cdn.jsdelivr.net/pyodide/v314.0.5/full/"
PLAYWRIGHT_TIMEOUT_MS = 0

HTML = """<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><title>pyodide numpy bench</title></head>
<body>
<pre id="log">loading pyodide…</pre>
<script type="module">
  const log = (t) => { console.log(t); };
  try {
    const { loadPyodide } = await import("%INDEX%pyodide.mjs");
    const pyodide = await loadPyodide({ indexURL: "%INDEX%" });
    await pyodide.loadPackage("numpy");
    const bench = await (await fetch("./bench.py")).text();
    const config = await (await fetch("./config.json")).text();
    pyodide.FS.mkdirTree("/home/pyodide");
    pyodide.FS.writeFile("/home/pyodide/config.json", config);
    pyodide.FS.writeFile("/home/pyodide/bench.py", bench);
    pyodide.setStdout({ batched: (s) => { for (const line of s.split("\\n")) if (line) log(line); } });
    pyodide.setStderr({ batched: (s) => { for (const line of s.split("\\n")) if (line) log(line); } });
    pyodide.runPython("import os; os.chdir('/home/pyodide')");
    const extra = pyodide.runPython(`
import json, sys
try:
    import pyodide as _p
    pv = getattr(_p, "__version__", "")
except Exception:
    pv = ""
json.dumps({"python": sys.version.split()[0], "pyodide_version": pv})
`);
    log("META/" + extra);
    await pyodide.runPythonAsync("import runpy; runpy.run_path('bench.py', run_name='__main__')");
    log("META/" + JSON.stringify({done: true, pyodide_rc: 0}));
    window.__bench_done = true;
  } catch (e) {
    log("META/" + JSON.stringify({done: false, error: String(e)}));
    window.__bench_done = true;
    throw e;
  }
</script>
</body>
</html>
""".replace("%INDEX%", PYODIDE_INDEX)


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
    for prefix in ("META/", "ROW/"):
        idx = text.find(prefix)
        if idx >= 0:
            payload = text[idx + len(prefix) :]
            try:
                return prefix[:-1], json.loads(payload)
            except json.JSONDecodeError:
                return prefix[:-1], payload
    return None, None


async def run_playwright(page_url: str, jsonl: Path):
    from playwright.async_api import async_playwright

    rows: list[dict] = []
    metas: list[dict] = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        page.set_default_timeout(PLAYWRIGHT_TIMEOUT_MS)

        def handle_console(msg):
            txt = msg.text
            kind, payload = parse_console_text(txt)
            if kind == "ROW" and isinstance(payload, dict):
                rows.append(payload)
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
            elif txt and "JSHandle" not in txt:
                print(txt, flush=True)

        page.on("console", handle_console)
        await page.goto(page_url)
        # Wait until bench emits done meta (or error).
        await page.wait_for_function(
            """() => window.__bench_done === true""",
            timeout=0,
        )
        version = browser.version
        await browser.close()
    return rows, metas, version


def serve(root: Path, port: int) -> ThreadingHTTPServer:
    class Quiet(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(root), **kwargs)

        def log_message(self, fmt, *args):
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", port), Quiet)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    return httpd


def keep_row(row: dict, max_n: int | None) -> bool:
    n = int(row.get("n") or 0)
    if n == 4096:
        return False
    if str(row.get("dtype") or "").startswith("complex"):
        return False
    if max_n is None:
        max_n = 1024
    if row.get("group") == "l1":
        return True
    return n <= max_n


def jsonl_to_csv(jsonl: Path, csv_path: Path, max_n: int | None = None) -> None:
    import csv

    rows = []
    if jsonl.exists():
        with jsonl.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    rows = [r for r in rows if keep_row(r, max_n)]
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
    p.add_argument("--max-n", type=int, default=1024)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--samples", type=int, default=10)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--skip-run", action="store_true")
    p.add_argument("--no-compare", action="store_true")
    args = p.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    jsonl = RESULTS / "pyodide.jsonl"

    if not args.skip_run:
        if not args.resume and jsonl.exists():
            jsonl.unlink()
        staging = HERE / "work" / "np-wasm-pyodide"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        shutil.copyfile(HERE / "bench.py", staging / "bench.py")
        skip = load_done(jsonl) if args.resume else []
        (staging / "config.json").write_text(
            json.dumps(
                {
                    "expect_blas": None,
                    "max_n": args.max_n,
                    "warmup": args.warmup,
                    "samples": args.samples,
                    "skip": skip,
                }
            )
        )
        (staging / "index.html").write_text(HTML)
        port = find_free_port()
        httpd = serve(staging, port)
        try:
            print(f"=== pyodide {PYODIDE_INDEX} max_n={args.max_n} ===", flush=True)
            rows, metas, chromium = asyncio.run(
                run_playwright(f"http://127.0.0.1:{port}/index.html", jsonl)
            )
        finally:
            httpd.shutdown()
        meta = {"pyodide_index": PYODIDE_INDEX, "label": "pyodide"}
        for m in metas:
            meta.update(m)
        meta["chromium"] = chromium
        (RESULTS / "pyodide.meta.json").write_text(json.dumps(meta, indent=2, default=str))
        if not any(m.get("done") for m in metas):
            raise RuntimeError("pyodide wasm run did not finish")

    jsonl_to_csv(jsonl, RESULTS / "pyodide.csv", max_n=args.max_n)
    jsonl_to_csv(RESULTS / "openblas.jsonl", RESULTS / "openblas_n1024.csv", max_n=args.max_n)
    if args.no_compare:
        return 0
    from compare import compare_and_report

    compare_and_report(
        RESULTS / "openblas_n1024.csv",
        RESULTS / "pyodide.csv",
        RESULTS / "openblas.meta.json" if (RESULTS / "openblas.meta.json").exists() else None,
        RESULTS / "pyodide.meta.json" if (RESULTS / "pyodide.meta.json").exists() else None,
        RESULTS / "compare_pyodide.csv",
        RESULTS / "report_pyodide.html",
    )
    from compare import write_combined_index
    write_combined_index(RESULTS)
    return 0


if __name__ == "__main__":
    os.chdir(HERE)
    raise SystemExit(main())
