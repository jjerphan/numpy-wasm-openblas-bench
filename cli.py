"""Pixi entry point: run labelled benches, HTML report, and blog figures."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PIXI_ENVS = HERE / ".pixi" / "envs"
RESULTS = HERE / "results"

CONFIGS = ("noblas", "ob034", "obdev", "obdev-relaxed", "pyodide", "cf64")


def _pixi_env_prefixes() -> dict[str, Path]:
    """Resolve env prefixes, including pixi detached-environments layouts."""
    try:
        info = subprocess.check_output(
            ["pixi", "info", "--json"],
            cwd=HERE,
            text=True,
        )
        import json

        envs = json.loads(info).get("environments_info") or []
        out = {}
        for env in envs:
            name = env.get("name")
            prefix = env.get("prefix")
            if name and prefix:
                out[name] = Path(prefix)
        return out
    except (OSError, subprocess.CalledProcessError, ValueError, KeyError):
        return {}


def require_env(name: str) -> Path:
    path = PIXI_ENVS / name
    if path.is_dir():
        return path
    detached = _pixi_env_prefixes().get(name)
    if detached is not None and detached.is_dir():
        return detached
    raise SystemExit(f"missing prefix for env {name!r}\nRun: pixi run setup")


def python_script(script: str, argv: list[str]) -> int:
    return subprocess.call([sys.executable, str(HERE / script), *argv])


def bench(config: str, full: bool, rest: list[str]) -> int:
    if config == "pyodide":
        return python_script("run_pyodide.py", ["--no-compare", *rest])
    if config == "cf64":
        env = require_env("cf64")
        return python_script(
            "run_native.py",
            [
                "--env",
                str(env),
                "--python",
                str(env / "bin" / "python"),
                "--label",
                "cf64",
                *rest,
            ],
        )
    env = require_env(config)
    argv = ["--label", config, "--no-compare", *rest]
    if config == "noblas":
        argv = [
            "--only",
            "noblas",
            "--noblas-env",
            str(env),
            *argv,
        ]
        if not full:
            argv.extend(["--skip-l3-n", "1024"])
    else:
        argv = [
            "--only",
            "openblas",
            "--openblas-env",
            str(env),
            *argv,
        ]
    return python_script("run_host.py", argv)


def convert_jsonls() -> None:
    from run_host import jsonl_to_csv

    for stem in CONFIGS:
        jsonl = RESULTS / f"{stem}.jsonl"
        if jsonl.exists():
            jsonl_to_csv(jsonl, RESULTS / f"{stem}.csv")


def report() -> int:
    from compare import write_combined_index

    RESULTS.mkdir(parents=True, exist_ok=True)
    convert_jsonls()
    write_combined_index(RESULTS)
    html = RESULTS / "report.html"
    if html.exists():
        print(f"report: {html}")
    return 0


def plot_blog(rest: list[str] | None = None) -> int:
    return python_script("plot_blog.py", list(rest or []))


def all_configs(full: bool, rest: list[str]) -> int:
    for config in CONFIGS:
        print(f"=== bench {config} ===", flush=True)
        rc = bench(config, full, rest)
        if rc != 0:
            return rc
    rc = report()
    if rc != 0:
        return rc
    print("=== plot-blog ===", flush=True)
    return plot_blog()


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # Forward plot-blog flags (e.g. --out) without treating them as a config.
    if argv and argv[0] == "plot-blog":
        return plot_blog(argv[1:])

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("bench", "report", "plot-blog", "all"))
    p.add_argument("config", nargs="?", choices=CONFIGS)
    p.add_argument(
        "--full",
        action="store_true",
        help="include no-BLAS Level-3 GEMM at n=1024 (slow)",
    )
    args, rest = p.parse_known_args(argv)
    if args.command == "report":
        return report()
    if args.command == "plot-blog":
        return plot_blog(rest)
    if args.command == "all":
        return all_configs(args.full, rest)
    if args.command == "bench":
        if not args.config:
            p.error("bench requires a config: " + ", ".join(CONFIGS))
        return bench(args.config, args.full, rest)
    p.error(args.command)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
