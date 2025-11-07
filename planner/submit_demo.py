#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
planner/submit_demo.py — fire demo jobs into the Fabric DT (local or remote).

Examples
--------
# 1 job, dry-run, local
python3 -m planner.submit_demo

# 25 jobs, reserve capacity, remote API
python3 -m planner.submit_demo --remote http://127.0.0.1:8080 -n 25 --no-dry-run

# 50 jobs, 5 workers, ~2 jobs/sec total, save CSV+JSON
python3 -m planner.submit_demo -n 50 -w 5 --qps 2.0 --out-json /tmp/demo.json --out-csv /tmp/demo.csv

# Single fixed demo job (3 stages), remote
python3 -m planner.submit_demo --remote http://127.0.0.1:8080 --fixed

What it does
------------
- Generates demo jobs with realistic stage shapes (IO → Preproc → Model → Post).
- Submits to local planner (imports DT modules) OR remote API (/plan).
- Prints summaries, tracks failures, and can export all results.
"""

from __future__ import annotations
import argparse
import json
import math
import os
import random
import string
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Pretty console output (optional)
try:
    from rich.console import Console
    from rich.table import Table
    from rich.progress import Progress, BarColumn, TimeElapsedColumn, TimeRemainingColumn, SpinnerColumn
    from rich import box
    RICH = True
    console = Console()
except Exception:
    RICH = False
    console = None  # type: ignore

# ---------------- Local DT imports (guarded) ----------------
def _try_import_local():
    try:
        from dt.state import DTState
        from dt.cost_model import CostModel
        from dt.policy.greedy import GreedyPlanner
        try:
            from dt.policy.bandit import BanditPolicy
        except Exception:
            BanditPolicy = None  # type: ignore
        return DTState, CostModel, GreedyPlanner, BanditPolicy
    except Exception:
        return None, None, None, None

# ---------------- Utilities ----------------

def _rand_id(prefix: str = "demo") -> str:
    s = "".join(random.choices(string.ascii_lowercase + string.digits, k=6))
    return f"{prefix}-{s}-{int(time.time()*1000)}"

def _tri(a: float, b: float, c: float) -> float:
    """Triangular distribution helper."""
    return random.triangular(a, c, b)

def _round(x: float, d: int = 2) -> float:
    return float(f"{x:.{d}f}")

def _pick_formats(node_like: Optional[Dict[str, Any]] = None) -> List[str]:
    # Typical formats in our world
    base = ["native", "wasm", "cuda", "npu"]
    # Bias towards native/wasm being present; cuda/npu sometimes
    fmts = []
    if random.random() < 0.95: fmts.append("native")
    if random.random() < 0.80: fmts.append("wasm")
    if random.random() < 0.60: fmts.append("cuda")
    if random.random() < 0.25: fmts.append("npu")
    return sorted(list(set(fmts)))

# ---------------- Demo job generation ----------------

def make_fixed_job() -> Dict[str, Any]:
    return {
        "id": _rand_id("fixed"),
        "deadline_ms": 4000,
        "stages": [
            {"id":"ingest","type":"io","size_mb":40,"resources":{"cpu_cores":1,"mem_gb":1},"allowed_formats":["native","wasm"]},
            {"id":"preproc","type":"preproc","size_mb":60,"resources":{"cpu_cores":2,"mem_gb":2},"allowed_formats":["native","cuda","wasm"]},
            {"id":"mlp","type":"mlp","size_mb":100,"resources":{"cpu_cores":4,"mem_gb":4,"gpu_vram_gb":2},"allowed_formats":["cuda","native"]}
        ]
    }

def make_random_job(stages_min: int = 3, stages_max: int = 6) -> Dict[str, Any]:
    k = random.randint(stages_min, stages_max)
    jid = _rand_id("job")
    ddl = int(_tri(2500, 8000, 4500))  # between ~2.5s and 8s, mode ~4.5s

    types = ["io", "preproc", "cv", "nlp", "analytics", "mlp", "viz"]
    stages = []
    size_acc = 0.0
    for i in range(k):
        t = random.choice(types)
        # Size grows then shrinks a bit toward the end
        size_mb = max(10, int(_tri(20, 200, 60) * (1.0 + 0.15 * math.sin(i))))
        size_acc += size_mb
        cpu = max(1, int(_tri(1, 8, 2 if t in ("io", "preproc") else 4)))
        mem = max(1, int(_tri(1, 16, 4)))
        vram = 0
        if t in ("cv", "mlp", "nlp") and random.random() < 0.6:
            vram = max(1, int(_tri(1, 16, 4)))

        allowed = _pick_formats()

        stages.append({
            "id": f"s{i+1}_{t}",
            "type": t,
            "size_mb": size_mb,
            "resources": {
                "cpu_cores": cpu,
                "mem_gb": mem,
                "gpu_vram_gb": vram
            },
            "allowed_formats": allowed
        })

    return {"id": jid, "deadline_ms": ddl, "stages": stages}

# ---------------- Submission engines ----------------

def submit_remote(base_url: str, job: Dict[str, Any], dry_run: bool = True, timeout: float = 30.0) -> Dict[str, Any]:
    import requests
    r = requests.post(
        base_url.rstrip("/") + "/plan",
        json={"job": job, "dry_run": dry_run},
        timeout=timeout
    )
    j = r.json()
    if not j.get("ok"):
        raise RuntimeError(f"remote /plan error: {j}")
    return j["data"]

def submit_local(job: Dict[str, Any], dry_run: bool = True) -> Dict[str, Any]:
    DTState, CostModel, GreedyPlanner, BanditPolicy = _try_import_local()
    if DTState is None:
        raise RuntimeError("Local DT modules not importable. Run from project root and ensure requirements installed.")

    state = DTState()
    cm = CostModel(state)
    bandit = BanditPolicy(persist_path="sim/bandit_state.json") if BanditPolicy else None
    planner = GreedyPlanner(state, cm, bandit=bandit, cfg={
        "risk_weight": 10.0,
        "energy_weight": 0.0,
        "prefer_locality_bonus_ms": 0.5,
        "require_format_match": False,
    })
    return planner.plan_job(job, dry_run=dry_run)

# ---------------- Result containers ----------------

@dataclass
class DemoResult:
    job_id: str
    latency_ms: float
    energy_kj: float
    risk: float
    infeasible: bool
    stages: List[Dict[str, Any]]

# ---------------- Rendering ----------------

def print_result(res: Dict[str, Any]):
    if not RICH:
        print(f"Job {res.get('job_id')}  latency={res.get('latency_ms')}ms  energy={res.get('energy_kj')}kJ  risk={res.get('risk')}  infeasible={res.get('infeasible')}")
        for s in (res.get("per_stage") or []):
            print(f"  - {s.get('id')} → {s.get('node')}  fmt={s.get('format')}  c={s.get('compute_ms')}ms  x={s.get('xfer_ms')}ms")
        return

    t = Table(title=f"Job {res.get('job_id')}", box=box.SIMPLE)
    t.add_column("Stage", style="bold")
    t.add_column("Node")
    t.add_column("Fmt", justify="center")
    t.add_column("Comp (ms)", justify="right")
    t.add_column("Xfer (ms)", justify="right")
    t.add_column("Risk", justify="right")
    for s in (res.get("per_stage") or []):
        t.add_row(
            str(s.get("id")),
            str(s.get("node") or "—"),
            str(s.get("format") or "—"),
            f"{_round(float(s.get('compute_ms', 0.0)),1)}",
            f"{_round(float(s.get('xfer_ms', 0.0)),1)}",
            f"{_round(float(s.get('risk', 0.0)),3)}",
        )
    if console:
        console.print(t)
        console.print(f"[b]Latency[/b]: {res.get('latency_ms')} ms    "
                      f"[b]Energy[/b]: {res.get('energy_kj')} kJ    "
                      f"[b]Risk[/b]: {res.get('risk')}    "
                      f"[b]Infeasible[/b]: {'YES' if res.get('infeasible') else 'no'}")

# ---------------- CSV/JSON export ----------------

def export_json(path: Union[str, Path], results: List[Dict[str, Any]]):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(results, indent=2), encoding="utf-8")

def export_csv(path: Union[str, Path], results: List[Dict[str, Any]]):
    import csv
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        # header
        w.writerow(["job_id","latency_ms","energy_kj","risk","infeasible","stage_id","node","format","compute_ms","xfer_ms"])
        for r in results:
            jid = r.get("job_id")
            lat = r.get("latency_ms")
            eng = r.get("energy_kj")
            risk = r.get("risk")
            inf = r.get("infeasible")
            for s in (r.get("per_stage") or []):
                w.writerow([jid, lat, eng, risk, inf,
                            s.get("id"), s.get("node"), s.get("format"),
                            s.get("compute_ms"), s.get("xfer_ms")])

# ---------------- Work queue (simple) ----------------

def throttle_sleep(qps: float, last_ts: List[float]):
    """Enforce global QPS by sleeping if needed. last_ts is a single-item list."""
    if qps <= 0: 
        return
    now = time.time()
    min_gap = 1.0 / qps
    if last_ts[0] is None:
        last_ts[0] = now
        return
    since = now - last_ts[0]
    if since < min_gap:
        time.sleep(min_gap - since)
    last_ts[0] = time.time()

def worker_submit(jobs: List[Dict[str, Any]], remote: Optional[str], dry_run: bool, qps: float, out_results: List[Dict[str, Any]], progress=None, task_id=None):
    last_ts = [None]
    for j in jobs:
        throttle_sleep(qps, last_ts)
        try:
            res = submit_remote(remote, j, dry_run=dry_run) if remote else submit_local(j, dry_run=dry_run)
        except Exception as e:
            res = {"job_id": j.get("id"), "error": str(e), "infeasible": True, "per_stage": [], "latency_ms": float("inf"), "energy_kj": 0.0, "risk": 1.0}
        out_results.append(res)
        if progress and task_id:
            progress.advance(task_id)
        if RICH:
            print_result(res)

# ---------------- Main ----------------

def main():
    ap = argparse.ArgumentParser(description="Submit many demo jobs to Fabric DT (local or remote)")
    ap.add_argument("--remote", default=None, help="Base URL of dt/api (e.g., http://127.0.0.1:8080)")
    ap.add_argument("-n", "--num", type=int, default=1, help="Number of demo jobs to submit")
    ap.add_argument("--fixed", action="store_true", help="Use a single fixed 3-stage demo job (else random jobs)")
    ap.add_argument("--stages-min", type=int, default=3, help="Min stages for random jobs")
    ap.add_argument("--stages-max", type=int, default=6, help="Max stages for random jobs")
    ap.add_argument("--no-dry-run", action="store_true", help="Reserve capacity (default: dry-run)")
    ap.add_argument("-w", "--workers", type=int, default=1, help="Concurrent workers (client-side)")
    ap.add_argument("--qps", type=float, default=0.0, help="Global rate limit (jobs per second). 0 = unlimited")
    ap.add_argument("--out-json", default=None, help="Write all results to JSON")
    ap.add_argument("--out-csv", default=None, help="Write all results to CSV (flat rows)")
    args = ap.parse_args()

    dry_run = not args.no_dry_run
    N = max(1, int(args.num))

    # Build jobs
    jobs: List[Dict[str, Any]] = []
    if args.fixed:
        for _ in range(N):
            jobs.append(make_fixed_job())
    else:
        for _ in range(N):
            jobs.append(make_random_job(args.stages_min, args.stages_max))

    # Submit
    results: List[Dict[str, Any]] = []

    if RICH and N > 1:
        total = N
        per_worker = math.ceil(N / max(1, args.workers))
        chunks = [jobs[i:i+per_worker] for i in range(0, N, per_worker)]
        with Progress(
            SpinnerColumn(),
            "[progress.description]{task.description}",
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console
        ) as progress:
            task = progress.add_task("[cyan]Submitting jobs...", total=total)
            for ch in chunks:
                worker_submit(ch, args.remote, dry_run, args.qps, results, progress, task)
    else:
        worker_submit(jobs, args.remote, dry_run, args.qps, results)

    # Exports
    if args.out_json:
        export_json(args.out_json, results)
        if RICH: console.print(f"[green]Saved JSON →[/green] {args.out_json}")  # type: ignore
        else: print(f"Saved JSON -> {args.out_json}")
    if args.out_csv:
        export_csv(args.out_csv, results)
        if RICH: console.print(f"[green]Saved CSV →[/green] {args.out_csv}")  # type: ignore
        else: print(f"Saved CSV -> {args.out_csv}")

    # Final summary
    if RICH:
        ok = sum(1 for r in results if not r.get("infeasible"))
        bad = len(results) - ok
        lat = [float(r.get("latency_ms", 0.0)) for r in results if r.get("latency_ms") not in (None, float("inf"))]
        if lat:
            p50 = sorted(lat)[len(lat)//2]
            p95 = sorted(lat)[int(len(lat)*0.95)-1 if len(lat)>1 else 0]
            console.print(f"\n[b]Jobs[/b]: {len(results)}  [green]ok[/green]: {ok}  [red]fail[/red]: {bad}  "
                          f"p50 latency: {int(p50)} ms  p95: {int(p95)} ms")
        else:
            console.print(f"\n[b]Jobs[/b]: {len(results)}  [green]ok[/green]: {ok}  [red]fail[/red]: {bad}")
    else:
        print(f"Jobs: {len(results)}  ok: {sum(1 for r in results if not r.get('infeasible'))}")

if __name__ == "__main__":
    main()

