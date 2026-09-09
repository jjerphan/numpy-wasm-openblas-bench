"""NumPy BLAS/LAPACK microbenchmark for wasm builds.

Covers every NumPy Python API that dispatches to CBLAS or LAPACK (not the
full CBLAS/LAPACK symbol catalogs). Reads optional config.json:
  expect_blas, max_n, warmup, samples, skip (list of [op, n, dtype]).

Prints META/<json> and ROW/<json>. Must not sys.exit (pyjs treats SystemExit
as failure).
"""

from __future__ import annotations

import json
import time
import traceback
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)

HERE = Path.cwd()
CONFIG_PATH = HERE / "config.json"

WARMUP = 5
SAMPLES = 10
SEED = 0
TARGET_S = 0.05
MAX_INNER = 20_000

REAL_DTYPES = (np.float32, np.float64)

# L2 / L3 / LAPACK share this grid. L1 uses vector lengths instead.
MATRIX_NS = (64, 128, 256, 512, 1024)
L1_NS = (10_000, 100_000, 1_000_000)
CONTROL_NS = (32, 64, 128, 256, 512)

CSV_FIELDS = (
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
)


def load_config() -> dict:
    cfg = {
        "expect_blas": None,
        "max_n": None,
        "warmup": WARMUP,
        "samples": SAMPLES,
        "skip": [],
    }
    if CONFIG_PATH.exists():
        cfg.update(json.loads(CONFIG_PATH.read_text()))
    return cfg


def dtype_name(dt) -> str:
    return np.dtype(dt).name


def is_complex(dt) -> bool:
    return np.issubdtype(np.dtype(dt), np.complexfloating)


def median(xs: list[float]) -> float:
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        return float("nan")
    mid = n // 2
    if n % 2:
        return xs[mid]
    return 0.5 * (xs[mid - 1] + xs[mid])


def gflops(flops: float | None, seconds: float) -> str:
    if flops is None or seconds <= 0 or seconds != seconds:
        return ""
    return f"{flops / seconds / 1e9:.6f}"


def emit_meta(obj: dict) -> None:
    print("META/" + json.dumps(obj, default=str), flush=True)


def emit_row(**kwargs) -> None:
    row = {k: kwargs.get(k, "") for k in CSV_FIELDS}
    print("ROW/" + json.dumps(row, default=str), flush=True)


def show_build() -> dict:
    cfg = np.show_config(mode="dicts")
    deps = cfg.get("Build Dependencies", {})
    blas = deps.get("blas", {})
    lapack = deps.get("lapack", {})
    return {
        "numpy_version": np.__version__,
        "blas_name": blas.get("name"),
        "blas_found": blas.get("found"),
        "blas_version": blas.get("version"),
        "blas_openblas_configuration": blas.get("openblas configuration"),
        "lapack_name": lapack.get("name"),
        "lapack_found": lapack.get("found"),
        "lapack_version": lapack.get("version"),
        "lapack_openblas_configuration": lapack.get("openblas configuration"),
    }


def gate_blas(info: dict, expect_blas) -> None:
    if expect_blas is None:
        return
    found = bool(info.get("blas_found"))
    name = str(info.get("blas_name") or "").lower()
    if expect_blas:
        if not found or "openblas" not in name:
            raise RuntimeError(
                f"expected OpenBLAS, got found={found} name={info.get('blas_name')!r}"
            )
    elif found:
        raise RuntimeError(
            f"expected no BLAS, got found={found} name={info.get('blas_name')!r}"
        )


def time_call(fn, warmup: int, samples: int) -> tuple[float, float, float, int]:
    t0 = time.perf_counter()
    fn()
    dt = time.perf_counter() - t0
    inner = 1 if dt >= TARGET_S else max(1, min(MAX_INNER, int(TARGET_S / max(dt, 1e-9))))
    for _ in range(warmup):
        for _ in range(inner):
            fn()
    times: list[float] = []
    for _ in range(samples):
        t0 = time.perf_counter()
        for _ in range(inner):
            fn()
        times.append((time.perf_counter() - t0) / inner)
    return median(times), min(times), max(times), inner


def run_one(
    op: str,
    n: int,
    dt,
    group: str,
    blas: str,
    setup,
    flops: float | None,
    warmup: int,
    samples: int,
    skip: set[tuple],
) -> None:
    key = (op, n, dtype_name(dt))
    if key in skip:
        return
    extra = dict(op=op, n=n, dtype=dtype_name(dt), group=group, blas=blas)
    try:
        fn = setup()
    except MemoryError:
        emit_row(**extra, inner="", median_s="", gflops="", min_s="", max_s="", status="oom")
        return
    except Exception as exc:
        emit_row(
            **extra,
            inner="",
            median_s="",
            gflops="",
            min_s="",
            max_s="",
            status=f"setup_error:{type(exc).__name__}:{exc}",
        )
        return
    try:
        med, lo, hi, inner = time_call(fn, warmup=warmup, samples=samples)
        emit_row(
            **extra,
            inner=inner,
            median_s=f"{med:.8e}",
            gflops=gflops(flops, med),
            min_s=f"{lo:.8e}",
            max_s=f"{hi:.8e}",
            status="ok",
        )
    except MemoryError:
        emit_row(**extra, inner="", median_s="", gflops="", min_s="", max_s="", status="oom")
    except Exception as exc:
        emit_row(
            **extra,
            inner="",
            median_s="",
            gflops="",
            min_s="",
            max_s="",
            status=f"error:{type(exc).__name__}:{exc}",
        )


def ns_ok(values, max_n):
    if max_n is None:
        return list(values)
    return [n for n in values if n <= max_n]


def randn(shape, dt, rng, order="C"):
    dt = np.dtype(dt)
    if np.issubdtype(dt, np.complexfloating):
        arr = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(dt)
    else:
        arr = rng.standard_normal(shape).astype(dt, copy=False)
    if order == "F":
        return np.asfortranarray(arr)
    return np.ascontiguousarray(arr)


def hpd(n, dt, rng):
    x = randn((n, n), dt, rng)
    return x @ x.conj().T + np.eye(n, dtype=dt) * n


def gemm_flops(n, dt) -> float:
    return (8.0 if is_complex(dt) else 2.0) * n * n * n


def gemv_flops(n, dt) -> float:
    return (8.0 if is_complex(dt) else 2.0) * n * n


def dot_flops(n, dt) -> float:
    return (8.0 if is_complex(dt) else 2.0) * n


def main() -> int:
    cfg = load_config()
    warmup = int(cfg.get("warmup", WARMUP))
    samples = int(cfg.get("samples", SAMPLES))
    max_n = cfg.get("max_n")
    skip = {tuple(x) for x in cfg.get("skip") or []}
    rng = np.random.default_rng(SEED)

    info = show_build()
    emit_meta(info)
    gate_blas(info, cfg.get("expect_blas"))

    matrix_ns = ns_ok(MATRIX_NS, max_n)
    control_ns = ns_ok(CONTROL_NS, max_n)
    has_vecdot = hasattr(np.linalg, "vecdot")
    has_vector_norm = hasattr(np.linalg, "vector_norm")
    has_matrix_norm = hasattr(np.linalg, "matrix_norm")
    has_svdvals = hasattr(np.linalg, "svdvals")

    def go(op, n, dt, group, blas, setup, flops):
        run_one(op, n, dt, group, blas, setup, flops, warmup, samples, skip)

    # --- Level 1 ---
    for dt in REAL_DTYPES:
        for n in L1_NS:
            fl = dot_flops(n, dt)

            def setup_dot(n=n, dt=dt):
                x = randn((n,), dt, rng)
                y = randn((n,), dt, rng)
                return lambda: np.dot(x, y)

            def setup_vdot(n=n, dt=dt):
                x = randn((n,), dt, rng)
                y = randn((n,), dt, rng)
                return lambda: np.vdot(x, y)

            def setup_inner(n=n, dt=dt):
                x = randn((n,), dt, rng)
                y = randn((n,), dt, rng)
                return lambda: np.inner(x, y)

            def setup_norm2(n=n, dt=dt):
                x = randn((n,), dt, rng)
                return lambda: np.linalg.norm(x, 2)

            go("dot", n, dt, "l1", "dot", setup_dot, fl)
            go("vdot", n, dt, "l1", "dotc", setup_vdot, fl)
            go("inner", n, dt, "l1", "dot", setup_inner, fl)
            go("norm2", n, dt, "l1", "nrm2", setup_norm2, fl)
            if has_vecdot:

                def setup_vecdot(n=n, dt=dt):
                    x = randn((n,), dt, rng)
                    y = randn((n,), dt, rng)
                    return lambda: np.linalg.vecdot(x, y)

                go("vecdot", n, dt, "l1", "dot", setup_vecdot, fl)
            if has_vector_norm:

                def setup_vnorm(n=n, dt=dt):
                    x = randn((n,), dt, rng)
                    return lambda: np.linalg.vector_norm(x)

                go("vector_norm", n, dt, "l1", "nrm2", setup_vnorm, fl)

    # --- Level 2 ---
    for dt in REAL_DTYPES:
        for n in matrix_ns:
            fl = gemv_flops(n, dt)

            def setup_gemv_n(n=n, dt=dt):
                a = randn((n, n), dt, rng)
                x = randn((n,), dt, rng)
                return lambda: a @ x

            def setup_gemv_t(n=n, dt=dt):
                a = randn((n, n), dt, rng)
                x = randn((n,), dt, rng)
                return lambda: x @ a

            go("gemv_n", n, dt, "l2", "gemv", setup_gemv_n, fl)
            go("gemv_t", n, dt, "l2", "gemv", setup_gemv_t, fl)

    # --- Level 3 ---
    for dt in REAL_DTYPES:
        for n in matrix_ns:
            fl = gemm_flops(n, dt)

            def setup_gemm(n=n, dt=dt):
                a = randn((n, n), dt, rng)
                b = randn((n, n), dt, rng)
                return lambda: a @ b

            def setup_gemm_f(n=n, dt=dt):
                a = randn((n, n), dt, rng, order="F")
                b = randn((n, n), dt, rng, order="F")
                return lambda: a @ b

            def setup_syrk(n=n, dt=dt):
                a = randn((n, n), dt, rng)
                return lambda: a @ a.T

            def setup_batch(n=n, dt=dt):
                a = randn((10, n, n), dt, rng)
                b = randn((10, n, n), dt, rng)
                return lambda: a @ b

            def setup_dot2d(n=n, dt=dt):
                a = randn((n, n), dt, rng)
                b = randn((n, n), dt, rng)
                return lambda: np.dot(a, b)

            def setup_matmul_fn(n=n, dt=dt):
                a = randn((n, n), dt, rng)
                b = randn((n, n), dt, rng)
                return lambda: np.matmul(a, b)

            def setup_tensordot(n=n, dt=dt):
                a = randn((n, n), dt, rng)
                b = randn((n, n), dt, rng)
                return lambda: np.tensordot(a, b, axes=1)

            def setup_multidot(n=n, dt=dt):
                a = randn((n, n), dt, rng)
                b = randn((n, n), dt, rng)
                c = randn((n, n), dt, rng)
                return lambda: np.linalg.multi_dot([a, b, c])

            def setup_mpow(n=n, dt=dt):
                a = randn((n, n), dt, rng)
                return lambda: np.linalg.matrix_power(a, 3)

            go("gemm", n, dt, "l3", "gemm", setup_gemm, fl)
            go("gemm_f", n, dt, "l3", "gemm", setup_gemm_f, fl)
            go("syrk", n, dt, "l3", "syrk", setup_syrk, fl)
            go("batched_matmul", n, dt, "l3", "gemm", setup_batch, 10.0 * fl)
            go("gemm_dot", n, dt, "l3", "gemm", setup_dot2d, fl)
            go("matmul", n, dt, "l3", "gemm", setup_matmul_fn, fl)
            go("tensordot", n, dt, "l3", "gemm", setup_tensordot, fl)
            go("multi_dot", n, dt, "l3", "gemm", setup_multidot, 2.0 * fl)
            go("matrix_power", n, dt, "l3", "gemm", setup_mpow, 2.0 * fl)

    # --- LAPACK ---
    for dt in REAL_DTYPES:
        for n in matrix_ns:

            def setup_solve(n=n, dt=dt):
                a = randn((n, n), dt, rng)
                b = randn((n,), dt, rng)
                return lambda: np.linalg.solve(a, b)

            def setup_inv(n=n, dt=dt):
                a = randn((n, n), dt, rng)
                return lambda: np.linalg.inv(a)

            def setup_chol(n=n, dt=dt):
                a = hpd(n, dt, rng)
                return lambda: np.linalg.cholesky(a)

            def setup_eigh(n=n, dt=dt):
                a = hpd(n, dt, rng)
                return lambda: np.linalg.eigh(a)

            def setup_eigvalsh(n=n, dt=dt):
                a = hpd(n, dt, rng)
                return lambda: np.linalg.eigvalsh(a)

            def setup_eig(n=n, dt=dt):
                a = randn((n, n), dt, rng)
                return lambda: np.linalg.eig(a)

            def setup_eigvals(n=n, dt=dt):
                a = randn((n, n), dt, rng)
                return lambda: np.linalg.eigvals(a)

            def setup_svd(n=n, dt=dt):
                a = randn((n, n), dt, rng)
                return lambda: np.linalg.svd(a, compute_uv=False)

            def setup_svd_full(n=n, dt=dt):
                a = randn((n, n), dt, rng)
                return lambda: np.linalg.svd(a, compute_uv=True)

            def setup_qr(n=n, dt=dt):
                a = randn((n, n), dt, rng)
                return lambda: np.linalg.qr(a)

            def setup_lstsq(n=n, dt=dt):
                a = randn((2 * n, n), dt, rng)
                b = randn((2 * n,), dt, rng)
                return lambda: np.linalg.lstsq(a, b, rcond=None)

            def setup_det(n=n, dt=dt):
                a = randn((n, n), dt, rng)
                return lambda: np.linalg.det(a)

            def setup_slogdet(n=n, dt=dt):
                a = randn((n, n), dt, rng)
                return lambda: np.linalg.slogdet(a)

            def setup_pinv(n=n, dt=dt):
                a = randn((n, n), dt, rng)
                return lambda: np.linalg.pinv(a)

            def setup_cond(n=n, dt=dt):
                a = randn((n, n), dt, rng)
                return lambda: np.linalg.cond(a)

            def setup_rank(n=n, dt=dt):
                a = randn((n, n), dt, rng)
                return lambda: np.linalg.matrix_rank(a)

            lapack_jobs = [
                ("solve", "gesv", setup_solve),
                ("inv", "getrf", setup_inv),
                ("cholesky", "potrf", setup_chol),
                ("eigh", "syevd", setup_eigh),
                ("eigvalsh", "syevd", setup_eigvalsh),
                ("eig", "geev", setup_eig),
                ("eigvals", "geev", setup_eigvals),
                ("svd", "gesdd", setup_svd),
                ("svd_full", "gesdd", setup_svd_full),
                ("qr", "geqrf", setup_qr),
                ("lstsq", "gelsd", setup_lstsq),
                ("det", "getrf", setup_det),
                ("slogdet", "getrf", setup_slogdet),
                ("pinv", "gesdd", setup_pinv),
                ("cond", "gesdd", setup_cond),
                ("matrix_rank", "gesdd", setup_rank),
            ]
            if has_svdvals:

                def setup_svdvals(n=n, dt=dt):
                    a = randn((n, n), dt, rng)
                    return lambda: np.linalg.svdvals(a)

                lapack_jobs.append(("svdvals", "gesdd", setup_svdvals))
            if has_matrix_norm:

                def setup_mnorm(n=n, dt=dt):
                    a = randn((n, n), dt, rng)
                    return lambda: np.linalg.matrix_norm(a, ord=2)

                lapack_jobs.append(("matrix_norm", "gesdd", setup_mnorm))
            for op, blas, setup in lapack_jobs:
                go(op, n, dt, "lapack", blas, setup, None)

            def setup_tsolve(n=n, dt=dt):
                a = randn((n, 2, n, 2), dt, rng)
                a = a + np.eye(n * 2, dtype=dt).reshape(n, 2, n, 2) * n
                b = randn((n, 2), dt, rng)
                return lambda: np.linalg.tensorsolve(a, b)

            def setup_tinv(n=n, dt=dt):
                a = randn((n, 2, n, 2), dt, rng)
                a = a + np.eye(n * 2, dtype=dt).reshape(n, 2, n, 2) * n
                return lambda: np.linalg.tensorinv(a)

            go("tensorsolve", n, dt, "lapack", "gesv", setup_tsolve, None)
            go("tensorinv", n, dt, "lapack", "getrf", setup_tinv, None)

    # --- Control ---
    for dt in REAL_DTYPES:
        for n in control_ns:

            def setup_sum(n=n, dt=dt):
                a = randn((n, n), dt, rng)
                return lambda: a.sum()

            go("sum", n, dt, "control", "", setup_sum, float(n * n))

    emit_meta({"done": True})
    return 0


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
