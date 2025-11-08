#!/usr/bin/env python3
"""
Monte-Carlo evaluator for Fabric.

- Loads node descriptors from ./nodes/*.yaml
- Loads one jobs YAML (DAG/pipeline) from ./jobs/<file>.yaml
- Optionally loads topology.yaml for link defaults
- For T trials:
    * Perturb node health/network
    * (Optionally) inject random failures
    * Obtain a placement either:
        - from DT endpoint (--dt http://.../plan), or
        - from a built-in greedy planner
    * Estimate per-stage times (compute + transfer), makespan, and SLO hit
- Outputs a CSV with per-trial metrics + prints summary (mean, p95, SLO hit-rate)

Usage examples:
  python3 sim/montecarlo.py --jobs jobs/jobs_10.yaml --trials 200
  python3 sim/montecarlo.py --jobs jobs/jobs_10.yaml --trials 200 --dt http://127.0.0.1:5055/plan
  python3 sim/montecarlo.py --jobs jobs/jobs_vision.yaml --topology sim/topology.yaml --seed 7 --trials 100

Assumptions/Notes:
- Greedy planner is intentionally simple but format/accelerator aware (CUDA/NPU/WASM/native).
- Compute time is a rough heuristic using CPU cores * GHz and an optional gpu.accel_score.
- Transfer time uses either explicit links (if topology present) or a default WAN/LAN guess.
- You can enhance cost_model parity with dt/cost_model.py later.
"""

import argparse, json, math, random, statistics, sys
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import yaml
from collections import defaultdict, Counter

try:
    import requests
except Exception:
    requests = None

# --------------------------
# IO helpers
# --------------------------

def load_yaml(p: Path) -> Any:
    with open(p, "r") as f:
        return yaml.safe_load(f)

def load_nodes(nodes_dir: Path) -> List[Dict[str, Any]]:
    nodes = []
    for f in sorted(nodes_dir.glob("*.yaml")):
        try:
            nodes.append(load_yaml(f))
        except Exception as e:
            print(f"[warn] could not load {f}: {e}")
    return nodes

def load_jobs(path: Path) -> Dict[str, Any]:
    return load_yaml(path)

def load_topology(path: Optional[Path]) -> Dict[str, Any]:
    if not path or not path.exists():
        return {}
    return load_yaml(path)


# --------------------------
# Synthetic workload generator
# --------------------------

SYNTHETIC_PROFILES: Dict[str, Dict[str, Any]] = {
    "cpu": {
        "size_mb": (10, 60),
        "cpu_cores": (1.0, 6.0),
        "mem_gb": (1.0, 8.0),
    },
    "gpu": {
        "size_mb": (40, 180),
        "cpu_cores": (1.0, 4.0),
        "mem_gb": (4.0, 12.0),
        "gpu_vram_gb": (2.0, 16.0),
        "allowed_formats": ["cuda", "native"],
    },
    "io": {
        "size_mb": (200, 600),
        "cpu_cores": (0.5, 2.0),
        "mem_gb": (2.0, 6.0),
    },
    "vision": {
        "size_mb": (80, 220),
        "cpu_cores": (2.0, 6.0),
        "mem_gb": (4.0, 10.0),
        "gpu_vram_gb": (4.0, 12.0),
        "allowed_formats": ["cuda", "npu", "wasm"],
    },
    "edge": {
        "size_mb": (5, 25),
        "cpu_cores": (0.5, 2.0),
        "mem_gb": (0.5, 2.0),
        "allowed_formats": ["wasm", "native"],
    },
}


def _rand_range(rng: Tuple[float, float]) -> float:
    return random.uniform(rng[0], rng[1])


def build_stage(profile: str, index: int) -> Dict[str, Any]:
    meta = SYNTHETIC_PROFILES.get(profile.lower()) or SYNTHETIC_PROFILES["cpu"]
    stage = {
        "id": f"stage-{index}",
        "size_mb": round(_rand_range(meta.get("size_mb", (10, 40))), 2),
        "resources": {
            "cpu_cores": round(_rand_range(meta.get("cpu_cores", (1.0, 3.0))), 2),
            "mem_gb": round(_rand_range(meta.get("mem_gb", (1.0, 4.0))), 2),
        },
    }
    if "gpu_vram_gb" in meta:
        stage["resources"]["gpu_vram_gb"] = round(_rand_range(meta["gpu_vram_gb"]), 2)
    if meta.get("allowed_formats"):
        stage["allowed_formats"] = meta["allowed_formats"]
    if profile.lower() == "edge":
        stage["labels"] = {"preferred_zone": "edge"}
    return stage


def _random_dag_edges(stage_ids: List[str]) -> List[Dict[str, str]]:
    edges: List[Dict[str, str]] = []
    for i, src in enumerate(stage_ids):
        for dst in stage_ids[i + 1 :]:
            if random.random() < 0.35:
                edges.append({"from": src, "to": dst})
    return edges


def generate_synthetic_catalog(
    count: int,
    profiles: List[str],
    max_stages: int,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    if seed is not None:
        random.seed(seed)

    catalog: List[Dict[str, Any]] = []
    profile_pool = [p for p in profiles if p]
    if not profile_pool:
        profile_pool = list(SYNTHETIC_PROFILES.keys())

    for idx in range(1, count + 1):
        num_stages = max(1, random.randint(2, max(2, max_stages)))
        stages: List[Dict[str, Any]] = []
        selected_profiles = random.choices(profile_pool, k=num_stages)
        for sidx, prof in enumerate(selected_profiles, start=1):
            stages.append(build_stage(prof, sidx))

        stage_ids = [st["id"] for st in stages]
        edges = _random_dag_edges(stage_ids)
        deadline = random.choice([0, random.randint(800, 2500), random.randint(2500, 6000)])

        job = {
            "id": f"synthetic-job-{idx}",
            "stages": stages,
            "deadline_ms": deadline,
            "edges": edges,
            "redundancy": random.choice([1, 1, 2, 3]),
            "priority": random.choice(["low", "normal", "high"]),
        }

        catalog.append(job)

    return {"jobs": catalog}


# --------------------------
# Simple link db (from topology)
# --------------------------

def link_key(a: str, b: str) -> str:
    return "|".join(sorted([a, b]))

def build_link_db(topology: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    db: Dict[str, Dict[str, float]] = {}
    if not topology:
        return db
    # link_profiles are not expanded here; links may carry everything inline
    for ln in topology.get("links", []) or []:
        a = ln.get("a"); b = ln.get("b")
        if not a or not b: 
            continue
        db[link_key(a,b)] = {
            "speed_gbps": float(ln.get("speed_gbps", 1.0)),
            "rtt_ms": float(ln.get("rtt_ms", 5.0)),
            "jitter_ms": float(ln.get("jitter_ms", 0.5)),
            "loss_pct": float(ln.get("loss_pct", 0.0)),
            "down": False
        }
    return db

def get_link_metrics(db: Dict[str, Dict[str, float]], a: str, b: str,
                     default_speed_gbps: float = 1.0,
                     default_rtt_ms: float = 5.0,
                     default_jitter_ms: float = 0.5) -> Tuple[float, float, float, float, bool]:
    k = link_key(a,b)
    if k in db:
        d = db[k]
        return d.get("speed_gbps", default_speed_gbps), d.get("rtt_ms", default_rtt_ms), d.get("jitter_ms", default_jitter_ms), d.get("loss_pct", 0.0), bool(d.get("down", False))
    # fall back to “regional” defaults if needed (simple)
    return default_speed_gbps, default_rtt_ms, default_jitter_ms, 0.0, False

# --------------------------
# Perturbations (Monte-Carlo)
# --------------------------

def sample_failure(prob: float) -> bool:
    return random.random() < max(0.0, min(1.0, prob))

def apply_perturbations(nodes: List[Dict[str, Any]],
                        link_db: Dict[str, Dict[str, float]],
                        node_fail_base: float = 0.03,
                        link_fail_base: float = 0.02) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, float]]]:
    """
    Returns deep-ish copies of nodes/link_db with trial-specific perturbations applied.
    """
    import copy
    N = [copy.deepcopy(n) for n in nodes]
    L = copy.deepcopy(link_db)

    # Node failures & derates
    for n in N:
        # crude failure prob influenced by last_week_crashes and health
        crashes = int(n.get("health", {}).get("last_week_crashes", 0))
        thermal = float(n.get("health", {}).get("thermal_derate", 0.0))
        trust = float(n.get("labels", {}).get("trust", "0.9"))
        base = node_fail_base + crashes*0.01 + thermal*0.1 + (0.05 if trust < 0.75 else 0.0)

        if sample_failure(base * 0.5):  # not too aggressive
            n.setdefault("_sim", {})["down"] = True
        else:
            # sample a fresh thermal derate/jitter onto compute speed
            dr = max(0.0, min(0.8, random.gauss(thermal, 0.05)))
            n.setdefault("_sim", {})["thermal_derate"] = dr

    # Link failures/degrades
    for k, d in L.items():
        base = link_fail_base + (d.get("loss_pct", 0.0)/100.0)*0.2
        if sample_failure(base * 0.5):
            d["down"] = True
        else:
            # degrade speed/jitter a bit
            d["speed_gbps"] = max(0.05, random.uniform(0.5, 1.0) * d.get("speed_gbps", 1.0))
            d["rtt_ms"] *= random.uniform(0.8, 1.5)
            d["jitter_ms"] *= random.uniform(1.0, 2.5)
    return N, L

# --------------------------
# Tiny greedy planner (fallback)
# --------------------------

def accel_multiplier(node: Dict[str, Any], stage: Dict[str, Any]) -> float:
    """
    Returns a compute speed multiplier (>1 faster) if node has a suitable accelerator
    for this stage; otherwise 1.0.
    """
    fmt_allow = set(stage.get("allowed_formats") or [])
    fmt_dis = set(stage.get("disallowed_formats") or [])
    fmts = set(node.get("formats_supported") or [])
    # Prefer CUDA if requested/allowed
    mult = 1.0
    if ("cuda" in fmts) and ("cuda" not in fmt_dis) and (not fmt_allow or "cuda" in fmt_allow):
        score = float((node.get("gpu") or {}).get("accel_score", 0.0) or 0.0)
        if score > 0:
            mult = max(mult, min(1.0 + score/10.0, 6.0))  # 1.0 .. ~6.0
    # NPU small boost for CV-ish stages
    if ("npu" in fmts) and ("npu" not in fmt_dis) and (not fmt_allow or "npu" in fmt_allow):
        tops = float((node.get("accelerators") or {}).get("npu_tops", 0) or 0)
        mult = max(mult, min(1.0 + tops/10.0, 3.0))      # up to ~3x
    return mult

def node_compute_capacity(node: Dict[str, Any]) -> float:
    cpu = node.get("cpu", {})
    cores = float(cpu.get("cores", 1))
    ghz = float(cpu.get("base_ghz", 1.0))
    derate = float(node.get("_sim", {}).get("thermal_derate", 0.0))
    down = bool(node.get("_sim", {}).get("down", False))
    if down:
        return 0.0
    return max(0.0, cores * ghz * (1.0 - derate))

def stage_compute_time_ms(stage: Dict[str, Any], node: Dict[str, Any]) -> float:
    cap = node_compute_capacity(node)
    if cap <= 0:
        return float("inf")
    # crude workload “size”: size_mb + cpu_cores request
    size_mb = float(stage.get("size_mb", 10.0))
    cpu_req = float((stage.get("resources") or {}).get("cpu_cores", 1.0))
    base = size_mb * 2.0 + cpu_req * 100.0   # arbitrary but consistent
    base = max(20.0, base)                  # min floor
    base /= max(1.0, cap/10.0)
    base /= accel_multiplier(node, stage)
    return base

def transfer_time_ms(src_node: str, dst_node: str, size_mb: float,
                     assign: Dict[str, str],
                     link_db: Dict[str, Dict[str, float]],
                     nodes_by_name: Dict[str, Dict[str, Any]]) -> float:
    if size_mb <= 0 or src_node == dst_node:
        return 0.0
    # Use link_db if site names match nodes; otherwise, approximate using node NICs
    # For now, treat node names as endpoints in link_db, else infer by labels.zone pairs.
    speed_gbps, rtt_ms, jitter_ms, loss_pct, down = get_link_metrics(
        link_db, src_node, dst_node, default_speed_gbps=1.0, default_rtt_ms=5.0
    )
    if down:
        return float("inf")
    # convert speed to MB/s
    mbps = speed_gbps * 1000.0
    eff_mbps = mbps * (1.0 - min(0.3, loss_pct/100.0)) * 0.85  # protocol overhead & loss penalty
    xfer = (size_mb * 8.0) / max(1.0, eff_mbps) * 1000.0       # ms
    return xfer + rtt_ms + jitter_ms

def greedy_place(job: Dict[str, Any], nodes: List[Dict[str, Any]], link_db: Dict[str, Dict[str, float]]) -> Dict[str, str]:
    """
    Very simple: place each stage on the node minimizing (compute_time + expected input transfer).
    Inputs: use job.datasets and stage.io.inputs if present; else ignore transfers.
    """
    assign: Dict[str, str] = {}
    nodes_by_name = {n["name"]: n for n in nodes}
    # For a linear pipeline (no edges): use stage order. If edges exist, topo-sort is needed; we do linear for MVP.
    stages: List[Dict[str, Any]] = job.get("stages") or []
    prev_node = None
    for st in stages:
        winners = []
        for n in nodes:
            if n.get("_sim", {}).get("down", False):
                continue
            # format gating (basic)
            allowed = set(st.get("allowed_formats") or [])
            disallowed = set(st.get("disallowed_formats") or [])
            fmts = set(n.get("formats_supported") or [])
            if allowed and not (fmts & allowed):
                continue
            if disallowed and (fmts & disallowed):
                continue
            # compute time on this node
            comp = stage_compute_time_ms(st, n)
            if prev_node:
                # assume st consumes previous stage output of size 'size_mb'
                xfer = transfer_time_ms(prev_node, n["name"], float(st.get("size_mb", 10.0)),
                                        assign, link_db, nodes_by_name)
            else:
                xfer = 0.0
            winners.append((comp + xfer, n["name"]))
        if not winners:
            # all infeasible -> pick any alive node to avoid crash
            fallback = next((nn["name"] for nn in nodes if not nn.get("_sim", {}).get("down", False)), None)
            assign[st["id"]] = fallback or "unavailable"
        else:
            winners.sort(key=lambda x: x[0])
            assign[st["id"]] = winners[0][1]
            prev_node = winners[0][1]
    return assign

def estimate_job_latency_ms(job: Dict[str, Any], assign: Dict[str, str],
                            nodes_by_name: Dict[str, Dict[str, Any]],
                            link_db: Dict[str, Dict[str, float]]) -> float:
    stages = job.get("stages") or []
    total = 0.0
    prev_node = None
    for st in stages:
        nid = assign.get(st["id"])
        n = nodes_by_name.get(nid) if nid else None
        if not n:
            return float("inf")
        # transfer from previous stage output to this node
        if prev_node:
            total += transfer_time_ms(prev_node, nid, float(st.get("size_mb", 10.0)), assign, link_db, nodes_by_name)
        # compute time
        total += stage_compute_time_ms(st, n)
        prev_node = nid
    return total

# --------------------------
# DT integration (optional)
# --------------------------

def plan_via_dt(dt_url: str, jobs_payload: Dict[str, Any],
                nodes: List[Dict[str, Any]],
                topology: Dict[str, Any]) -> Optional[Dict[str, str]]:
    if not requests:
        return None
    try:
        payload = {
            "jobs": jobs_payload,
            "nodes": nodes,
            "topology": topology,
            "mode": "montecarlo"
        }
        r = requests.post(dt_url, json=payload, timeout=5.0)
        if r.status_code != 200:
            return None
        data = r.json()
        # Expect {"assignments": {"<stage_id>": "<node_name>", ...}}
        return data.get("assignments")
    except Exception:
        return None

# --------------------------
# CSV writer
# --------------------------

def write_csv(path: Path, rows: List[Dict[str, Any]]):
    import csv
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        for r in rows:
            w.writerow(r)

# --------------------------
# Main
# --------------------------

def main():
    ap = argparse.ArgumentParser(description="Fabric Monte-Carlo evaluator")
    ap.add_argument("--nodes", type=str, default="nodes", help="Path to nodes/ dir")
    ap.add_argument("--jobs", type=str, help="Path to a jobs YAML (e.g., jobs/jobs_10.yaml)")
    ap.add_argument("--synthetic", type=int, default=0, help="Generate N synthetic jobs instead of reading a file")
    ap.add_argument("--synthetic-max-stages", type=int, default=4, help="Maximum number of stages per synthetic job")
    ap.add_argument(
        "--synthetic-profiles",
        type=str,
        default="cpu,gpu,vision,edge",
        help="Comma separated stage profiles to sample (cpu,gpu,vision,edge,io)",
    )
    ap.add_argument("--topology", type=str, default=None, help="Path to topology.yaml")
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--dt", type=str, default=None, help="Optional DT plan endpoint (e.g., http://127.0.0.1:5055/plan)")
    ap.add_argument("--out", type=str, default="sim/montecarlo_results.csv")
    args = ap.parse_args()

    random.seed(args.seed)

    nodes = load_nodes(Path(args.nodes))
    if not nodes:
        print("[error] no nodes found in ./nodes/")
        sys.exit(2)
    if args.synthetic > 0:
        profiles = [p.strip() for p in args.synthetic_profiles.split(",")]
        jobs_payload = generate_synthetic_catalog(
            args.synthetic,
            profiles,
            max_stages=max(1, args.synthetic_max_stages),
            seed=args.seed,
        )
        print(f"[synthetic] generated {args.synthetic} jobs using profiles {profiles}")
    elif args.jobs:
        jobs_payload = load_jobs(Path(args.jobs))
    else:
        print("[error] provide --jobs file or --synthetic count")
        sys.exit(2)
    topology = load_topology(Path(args.topology)) if args.topology else {}
    link_db_base = build_link_db(topology)

    # Prepare jobs list (the schema allows multiple jobs; run them sequentially here)
    jobs_list = jobs_payload.get("jobs") or []
    if not jobs_list:
        print("[error] job catalog is empty.")
        sys.exit(2)

    rows: List[Dict[str, Any]] = []
    slo_hits = 0
    latencies: List[float] = []

    for t in range(1, args.trials + 1):
        # Perturb for this trial
        nodes_trial, link_db_trial = apply_perturbations(nodes, link_db_base)

        # For each job, obtain placement and estimate time
        trial_total = 0.0
        trial_slo_ok = True
        combined_deadline = 0.0

        for job in jobs_list:
            # plan
            assign = None
            if args.dt:
                assign = plan_via_dt(args.dt, {"jobs": [job]}, nodes_trial, topology)
            if not assign:
                assign = greedy_place(job, nodes_trial, link_db_trial)

            # estimate
            nodes_by_name = {n["name"]: n for n in nodes_trial}
            lat_ms = estimate_job_latency_ms(job, assign, nodes_by_name, link_db_trial)
            trial_total += lat_ms
            deadline = float(job.get("deadline_ms", 0.0))
            combined_deadline += deadline
            ok = (deadline <= 0.0) or (lat_ms <= deadline)
            trial_slo_ok = trial_slo_ok and ok

        latencies.append(trial_total)
        if trial_slo_ok:
            slo_hits += 1

        rows.append({
            "trial": t,
            "latency_ms": round(trial_total, 2),
            "slo_ok": int(trial_slo_ok)
        })

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    write_csv(outp, rows)

    mean_lat = statistics.fmean(latencies)
    p95 = float(sorted(latencies)[max(0, int(math.ceil(0.95*len(latencies))-1))])
    p99 = float(sorted(latencies)[max(0, int(math.ceil(0.99*len(latencies))-1))])
    hit_rate = slo_hits / max(1, args.trials)

    print(f"[montecarlo] trials={args.trials}")
    print(f"[montecarlo] mean_ms={mean_lat:.2f}  p95_ms={p95:.2f}  p99_ms={p99:.2f}")
    print(f"[montecarlo] SLO_hit_rate={hit_rate*100:.1f}%")
    print(f"[montecarlo] wrote CSV → {outp.as_posix()}")

if __name__ == "__main__":
    main()

