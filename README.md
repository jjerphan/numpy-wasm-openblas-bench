# numpy-wasm-openblas-bench

Private harness: the same NumPy BLAS/LAPACK API grid in Chromium against
emscripten-forge wasm envs (with or without OpenBLAS), against Pyodide, and
against conda-forge linux-64 NumPy + OpenBLAS.

Requires **linux-64**, [pixi](https://pixi.sh/), and enough disk for Chromium.

## One-liners

```bash
pixi run setup             # install all prefixes + Chromium (once)
pixi run bench-noblas      # emscripten-forge NumPy, no OpenBLAS
pixi run bench-ob034       # NumPy 2.5.3 + OpenBLAS 0.3.34 (wasm)
pixi run bench-obdev       # same NumPy + OpenBLAS 0.3.35.dev0 (wasm)
pixi run bench-pyodide     # Pyodide 314.0.5 from the CDN
pixi run bench-cf64        # conda-forge linux-64, one thread, taskset
pixi run report            # results/report.html from existing CSVs
pixi run all               # sequential benches, then report
```

Flags after `--` go to the runner, e.g. `pixi run bench-ob034 -- --resume --max-n 512`.

`pixi run all` is sequential (do not overlap Playwright Chromium runs). It is still long: wasm packing plus OpenBLAS at `n = 1024`. By default **no-BLAS skips Level-3 GEMM at n = 1024**. Pass `--full` to include those cells:

```bash
pixi run bench-noblas -- --full
pixi run all -- --full
```

The HTML report is self-contained (no CDN). Missing CSVs are skipped, so `pixi run report` works after a subset of benches.

## Labels

| Task | Prefix | CSV stem |
| --- | --- | --- |
| `bench-noblas` | `.pixi/envs/noblas` | `noblas` |
| `bench-ob034` | `.pixi/envs/ob034` | `ob034` |
| `bench-obdev` | `.pixi/envs/obdev` | `obdev` |
| `bench-pyodide` | (CDN) | `pyodide` |
| `bench-cf64` | `.pixi/envs/cf64` | `cf64` |

## Scripts

- `cli.py` — pixi dispatcher (`bench`, `report`, `all`)
- `bench.py` — timed NumPy calls; prints `META/` and `ROW/` JSON. Must not `sys.exit`.
- `run_host.py` — `pyjs-code-runner` + Playwright against a wasm conda env
- `run_pyodide.py` — official Pyodide NumPy from the CDN
- `run_native.py` — conda-forge linux-64 NumPy + OpenBLAS, one thread, `taskset`
- `compare.py` / `report.py` — CSV compare and self-contained HTML
- `plot_blog.py` — SVG figures from labelled CSVs

Jsonl flushes per row. `--resume` continues a jsonl.
