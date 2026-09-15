# numpy-wasm-openblas-bench

Reproduce the NumPy BLAS/LAPACK API grid used in the
[faster NumPy in the browser](https://emscripten-forge.org/) blog post:
Chromium against emscripten-forge wasm envs (with or without OpenBLAS),
against Pyodide, and against conda-forge linux-64 NumPy + OpenBLAS.

Requires **linux-64**, [pixi](https://pixi.sh/), and enough disk for Chromium
and empack caches.

## One-liners

```bash
pixi run setup   # once: install all prefixes + Chromium
pixi run all     # all benches → results/report.html → results/blog/*
```

Flags after `--` go to the runners, for example:

```bash
pixi run bench-ob034 -- --resume --max-n 512
pixi run all -- --full          # include no-BLAS Level-3 GEMM at n=1024
```

`pixi run all` is sequential (do not overlap Playwright Chromium runs). It is
long: wasm packing plus OpenBLAS at `n = 1024`. By default **no-BLAS skips
Level-3 GEMM at n = 1024**; pass `--full` to include those cells.

Partial runs:

```bash
pixi run bench-noblas
pixi run bench-ob034
pixi run bench-obdev
pixi run bench-obdev-relaxed
pixi run bench-pyodide
pixi run bench-cf64
pixi run report       # results/report.html from existing CSVs
pixi run plot-blog    # results/blog/*.svg + appendix_tables.md
```

Committed `results/*.csv` and `results/blog/*` are reference outputs from the
blog measurements; regenerating overwrites them. Missing CSVs are skipped by
`report`, so you can run a subset of benches first.

## Configs

| Task | Prefix / source | CSV stem | Notes |
| --- | --- | --- | --- |
| `bench-noblas` | `.pixi/envs/noblas` | `noblas` | NumPy 2.5.2, no OpenBLAS |
| `bench-ob034` | `.pixi/envs/ob034` | `ob034` | NumPy 2.5.3 + OpenBLAS 0.3.34 ([recipes#6310](https://github.com/emscripten-forge/recipes/pull/6310)) |
| `bench-obdev` | `.pixi/envs/obdev` | `obdev` | same NumPy + OpenBLAS `0.3.35.dev0` `simd128_hbf81ecf_3` ([recipes#6761](https://github.com/emscripten-forge/recipes/pull/6761)) |
| `bench-obdev-relaxed` | `.pixi/envs/obdev-relaxed` | `obdev-relaxed` | OpenBLAS `0.3.35.dev0` `relaxed_simd_h13a9a80_3` ([recipes#6761](https://github.com/emscripten-forge/recipes/pull/6761)) |
| `bench-pyodide` | Pyodide CDN | `pyodide` | official Pyodide NumPy |
| `bench-cf64` | `.pixi/envs/cf64` | `cf64` | conda-forge linux-64, one thread, `taskset` |

OpenBLAS 0.3.35 packages come from
[`emscripten-forge-4x-experimental`](https://prefix.dev/channels/emscripten-forge-4x-experimental)
on prefix.dev (no local channel required).

## Scripts

- `cli.py` — pixi dispatcher (`bench`, `report`, `plot-blog`, `all`)
- `bench.py` — timed NumPy calls in the guest; prints `META/` and `ROW/` JSON. Must not call `sys.exit`.
- `run_host.py` — `pyjs-code-runner` + Playwright against a wasm conda env
- `run_pyodide.py` — official Pyodide NumPy from the CDN
- `run_native.py` — conda-forge linux-64 NumPy + OpenBLAS, one thread, `taskset`
- `compare.py` / `report.py` — CSV compare and self-contained HTML
- `plot_blog.py` — SVG figures and appendix tables under `results/blog/` (`--out`, optional `--copy-to`)

Jsonl flushes per row. `--resume` continues a jsonl.

## License

BSD-3-Clause. See [LICENSE](LICENSE).
