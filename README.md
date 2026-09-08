# numpy-wasm-openblas-bench

Private harness: run the same NumPy BLAS/LAPACK API grid in Chromium against
emscripten-forge wasm envs (with or without OpenBLAS) and against Pyodide.

## Scripts

- `bench.py` — timed NumPy calls; prints `META/` and `ROW/` JSON. Must not `sys.exit`.
- `run_host.py` — `pyjs-code-runner` + Playwright against a wasm conda env.
- `run_pyodide.py` — official Pyodide NumPy from the CDN.
- `compare.py` / `report.py` — CSV compare and a self-contained HTML report.
- `plot_blog.py` — SVG figures from labelled CSVs (`noblas`, `ob034`, `obdev`, `pyodide`).

## Host

A linux-64 env with `pyjs_code_runner` and Playwright Chromium, for example:

```bash
micromamba create -n np-wasm-host pyjs_code_runner playwright python
playwright install chromium
```

Wasm envs need `--platform=emscripten-wasm32` (NumPy, `pyjs`, optional `openblas`).

## Examples

```bash
micromamba run -n np-wasm-host python run_host.py \
  --only noblas --noblas-env /path/to/np-wasm-noblas \
  --max-n 1024 --no-compare

micromamba run -n np-wasm-host python run_host.py \
  --only openblas --openblas-env /path/to/np-wasm-ob034 \
  --label ob034 --max-n 1024 --no-compare

micromamba run -n np-wasm-host python run_pyodide.py --max-n 1024 --no-compare

python plot_blog.py
```

`--resume` continues a jsonl. `--max-n 1024` GEMM without BLAS is slow (hours possible).
Jsonl flushes per row. Do not overlap Playwright Chromium runs.
