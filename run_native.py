"""Run bench.py on conda-forge linux-64 NumPy + OpenBLAS (one thread, P-core)."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
from ctypes import CDLL, c_char_p
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
DEFAULT_ENV = HERE / ".pixi" / "envs" / "cf64"
WORK = HERE / "work" / "native"

THREAD_ENV = {
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "PYTHONNOUSERSITE": "1",
}


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


def parse_line(text: str) -> tuple[str | None, object]:
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


def overlay_runtime_openblas(conda_env: Path, meta: dict) -> None:
    """Replace NumPy show_config BLAS fields with the runtime OpenBLAS package."""
    meta["numpy_show_config_blas_name"] = meta.get("blas_name")
    meta["numpy_show_config_blas_version"] = meta.get("blas_version")
    meta["numpy_show_config_openblas_configuration"] = meta.get("blas_openblas_configuration")
    pc = conda_env / "lib" / "pkgconfig" / "openblas.pc"
    if not pc.exists():
        pc = conda_env / "lib" / "pkgconfig" / "blas.pc"
    if pc.exists():
        for line in pc.read_text().splitlines():
            if line.startswith("version="):
                meta["blas_version"] = line.split("=", 1)[1].strip()
                meta["lapack_version"] = meta["blas_version"]
            elif line.startswith("openblas_config="):
                meta["blas_openblas_configuration"] = line.split("=", 1)[1].strip()
                meta["lapack_openblas_configuration"] = meta["blas_openblas_configuration"]
    meta_dir = conda_env / "conda-meta"
    for glob in ("libopenblas-*.json", "openblas-*.json"):
        for p in sorted(meta_dir.glob(glob)):
            rec = json.loads(p.read_text())
            ver = rec.get("version")
            meta["openblas_conda"] = f"{ver}={rec.get('build')}"
            meta["openblas_channel"] = rec.get("channel")
            meta["openblas_conda_pkg"] = p.stem
            if ver:
                meta["blas_version"] = ver
                meta["lapack_version"] = ver
            break
        else:
            continue
        break
    so = conda_env / "lib" / "libopenblas.so.0"
    if so.exists() and not meta.get("blas_openblas_configuration"):
        try:
            lib = CDLL(str(so))
            lib.openblas_get_config.restype = c_char_p
            cfg = lib.openblas_get_config()
            if cfg:
                text = cfg.decode() if isinstance(cfg, bytes) else str(cfg)
                meta["blas_openblas_configuration"] = text
                meta["lapack_openblas_configuration"] = text
        except OSError:
            pass
    meta["blas_name"] = "openblas"
    meta["lapack_name"] = "openblas"


def machine_info() -> dict:
    cpu = ""
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        cpu = platform.processor()
    mem_kib = None
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                mem_kib = int(line.split()[1])
                break
    except OSError:
        pass
    os_pretty = ""
    try:
        for line in Path("/etc/os-release").read_text().splitlines():
            if line.startswith("PRETTY_NAME="):
                os_pretty = line.split("=", 1)[1].strip().strip('"')
                break
    except OSError:
        os_pretty = platform.platform()
    product = ""
    try:
        product = Path("/sys/devices/virtual/dmi/id/product_name").read_text().strip()
    except OSError:
        pass
    return {
        "cpu_model": cpu,
        "mem_gib": round(mem_kib / (1024 * 1024), 1) if mem_kib else None,
        "os": os_pretty,
        "product": product,
        "platform": platform.platform(),
    }


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


def run_native(
    python: Path,
    conda_env: Path,
    jsonl: Path,
    max_n: int | None,
    warmup: int,
    samples: int,
    resume: bool,
    cpu: int,
) -> dict:
    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True)
    shutil.copyfile(HERE / "bench.py", WORK / "bench.py")
    skip = load_done(jsonl) if resume else []
    (WORK / "config.json").write_text(
        json.dumps(
            {
                "expect_blas": None,
                "max_n": max_n,
                "warmup": warmup,
                "samples": samples,
                "skip": skip,
            }
        )
    )

    env = os.environ.copy()
    env.update(THREAD_ENV)
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    cmd = [
        "taskset",
        "-c",
        str(cpu),
        str(python),
        "-u",
        str(WORK / "bench.py"),
    ]
    print(f"=== native {python} cpu={cpu} threads=1 ===", flush=True)
    proc = subprocess.Popen(
        cmd,
        cwd=WORK,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    metas: list[dict] = []
    for line in proc.stdout:
        kind, payload = parse_line(line)
        if kind == "ROW" and isinstance(payload, dict):
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
        elif line.strip():
            print(line, end="", flush=True)
    rc = proc.wait()
    meta: dict = {}
    for m in metas:
        meta.update(m)
    meta["label"] = "cf64"
    meta["conda_env"] = str(conda_env)
    meta["python"] = str(python)
    meta["return_code"] = rc
    meta["openblas_num_threads"] = 1
    meta["openblas_runtime_num_threads"] = 1
    meta["cpu_affinity"] = cpu
    meta.update(machine_info())
    overlay_runtime_openblas(conda_env, meta)
    jsonl.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2, default=str))
    if rc != 0 and not meta.get("done"):
        raise RuntimeError(f"native run failed rc={rc}")
    return meta


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--env", type=Path, default=DEFAULT_ENV)
    p.add_argument("--python", type=Path, default=None)
    p.add_argument("--label", default="cf64")
    p.add_argument("--max-n", type=int, default=1024)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--samples", type=int, default=10)
    p.add_argument("--cpu", type=int, default=0, help="taskset CPU id (P-core)")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--skip-run", action="store_true")
    args = p.parse_args()

    python = args.python or (args.env / "bin" / "python")
    if not python.exists():
        raise SystemExit(f"missing python: {python}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    jsonl = RESULTS / f"{args.label}.jsonl"
    if not args.skip_run:
        if not args.resume and jsonl.exists():
            jsonl.unlink()
        run_native(
            python,
            args.env,
            jsonl,
            args.max_n,
            args.warmup,
            args.samples,
            args.resume,
            args.cpu,
        )
    if jsonl.exists():
        jsonl_to_csv(jsonl, RESULTS / f"{args.label}.csv")
    return 0


if __name__ == "__main__":
    os.chdir(HERE)
    raise SystemExit(main())
