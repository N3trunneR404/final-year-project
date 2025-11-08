#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
planner/run_plan.py — submit jobs to the Fabric DT (local or remote).

Usage
-----
# Local (imports dt.* directly)
python3 -m planner.run_plan --job jobs/jobs_10.yaml --dry-run

# Remote (use if dt/api.py is running on another process/machine)
python3 -m planner.run_plan --remote http://127.0.0.1:8080 --job jobs/jobs_10.yaml --dry-run

# Single job file, save result JSON
python3 -m planner.run_plan --job jobs/demo.yaml --out /tmp/plan.json

Options
-------
--job PATH            YAML file containing a single job object or a list under 'jobs'
--dry-run             Plan without reserving capacity (default: True)
--strategy STR        planner strategy (greedy, cheapest-energy, bandit, rl-markov, resilient, ...)
--remote URL          If provided, POSTs to {URL}/plan or {URL}/plan_batch
--repeat N            Repeat planning N times (useful for Monte-Carlo learning with bandit; local mode)
--out PATH            Save full JSON result(s) here
"""

from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

# Optional pretty console
try:
    from rich.console import Console
    from rich.table import Table
    RICH = True
    console = Console()
except Exception:
    RICH = False
    console = None  # type: ignore

# Local DT imports (guarded so remote mode can run without deps)
def _try_import_local():
    try:
        from dt.state import DTState
        from dt.cost_model import CostModel
        from dt.policy.greedy import GreedyPlanner
        from dt.policy.resilient import FederatedPlanner
        from dt.policy.mdp import MarkovPlanner
        from dt.policy.rl_stub import RLPolicy
        try:
            from dt.policy.bandit import BanditPolicy
        except Exception:
            BanditPolicy = None  # type: ignore
        return DTState, CostModel, GreedyPlanner, FederatedPlanner, MarkovPlanner, RLPolicy, BanditPolicy
    except Exception:
        return None, None, None, None, None, None, None

def load_yaml(path: Union[str, Path]) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def ensure_jobs(obj: Any) -> List[Dict[str, Any]]:
    # Accept either:
    #  - {id, stages[]} (single job)
    #  - {"jobs": [ ... ]}
    #  - [ {...}, {...} ]
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        if "jobs" in obj and isinstance(obj["jobs"], list):
            return obj["jobs"]
        # assume single job object
        return [obj]
    raise ValueError("YAML must be a job object, a list of jobs, or a dict with 'jobs': [...]")

def print_summary(results: List[Dict[str, Any]]):
    def fmt_ratio(val: Optional[float]) -> str:
        if val is None:
            return "—"
        try:
            return f"{float(val):.2f}"
        except Exception:
            return str(val)

    def fmt_pct(val: Optional[float]) -> str:
        if val is None:
            return "—"
        try:
            return f"{float(val) * 100:.0f}%"
        except Exception:
            return str(val)

    if not RICH:
        # Minimal stdout
        for r in results:
            feasible = not r.get("infeasible")
            spread = r.get("federation_spread")
            resilience = r.get("resilience_score")
            cross = r.get("cross_federation_fallback_ratio")
            avg_rel = r.get("avg_reliability")
            print(
                f"\nJob {r.get('job_id')}: latency={r.get('latency_ms')}ms  energy={r.get('energy_kj')}kJ  risk={r.get('risk')}"
                f" feasible={feasible} spread={fmt_ratio(spread)} resilience={fmt_pct(resilience)} cross_fallback={fmt_pct(cross)}"
                f" reliability={fmt_ratio(avg_rel)}"
            )
            for s in (r.get("per_stage") or []):
                fallbacks = s.get("fallbacks") or []
                fallback_feds = s.get("fallback_federations") or []
                fallback_tokens: List[str] = []
                if isinstance(fallbacks, list):
                    for entry in fallbacks:
                        if isinstance(entry, dict):
                            node = entry.get("node") or "?"
                            rel = entry.get("reliability")
                            score = entry.get("score")
                            token = node
                            if rel is not None:
                                token += f"(rel={float(rel):.2f})"
                            if score is not None:
                                token += f"@{float(score):.1f}"
                            fallback_tokens.append(token)
                        else:
                            fallback_tokens.append(str(entry))
                fallback_pairs: List[str] = []
                if fallback_feds:
                    for idx, fed in enumerate(fallback_feds):
                        node_token = fallback_tokens[idx] if idx < len(fallback_tokens) else "?"
                        fallback_pairs.append(f"{node_token}({fed})" if fed else node_token)
                fallback_str = f" fallback={','.join(fallback_tokens)}" if fallback_tokens else ""
                if fallback_pairs and fallback_pairs != fallback_tokens:
                    fallback_str += f" fallback_fed={','.join(fallback_pairs)}"
                load_term = s.get("load_factor")
                net_term = s.get("network_penalty")
                exp_cost = s.get("expected_cost")
                reliability = s.get("reliability")
                availability = s.get("availability_window_sec")
                extras = []
                if load_term is not None:
                    extras.append(f"load={fmt_ratio(load_term)}")
                if net_term is not None:
                    extras.append(f"net={fmt_ratio(net_term)}")
                if exp_cost is not None:
                    try:
                        extras.append(f"J={float(exp_cost):.3f}")
                    except Exception:
                        pass
                if reliability is not None:
                    extras.append(f"rel={fmt_ratio(reliability)}")
                if availability is not None:
                    extras.append(f"avail={fmt_ratio(availability)}s")
                extra_str = f" ({', '.join(extras)})" if extras else ""
                print(
                    f"  - {s.get('id')} → {s.get('node')}  fmt={s.get('format')}  c={s.get('compute_ms')}ms  x={s.get('xfer_ms')}ms  risk={s.get('risk')}{extra_str}{fallback_str}"
                )
        return

    # Pretty table
    tbl = Table(title="Plan Summary", show_lines=False)
    tbl.add_column("Job", style="bold")
    tbl.add_column("Latency (ms)", justify="right")
    tbl.add_column("Energy (kJ)", justify="right")
    tbl.add_column("Risk", justify="right")
    tbl.add_column("Reliability", justify="right")
    tbl.add_column("Spread", justify="right")
    tbl.add_column("Fallback", justify="right")
    tbl.add_column("Cross-fed", justify="right")
    tbl.add_column("Feasible", justify="center")
    tbl.add_column("Stages", style="dim")

    for r in results:
        stages_lines = []
        for s in (r.get("per_stage") or []):
            base = (
                f"[dim]{s.get('id')}[/dim] → [b]{s.get('node') or '—'}[/b] "
                f"{('['+str(s.get('format'))+']') if s.get('format') else ''} "
                f"(c:{s.get('compute_ms')}ms, x:{s.get('xfer_ms')}ms)"
            )
            extras = []
            if s.get("load_factor") is not None:
                extras.append(f"load={fmt_ratio(s.get('load_factor'))}")
            if s.get("network_penalty") is not None:
                extras.append(f"net={fmt_ratio(s.get('network_penalty'))}")
            if s.get("expected_cost") is not None:
                try:
                    extras.append(f"J={float(s.get('expected_cost')):.3f}")
                except Exception:
                    pass
            if s.get("reliability") is not None:
                extras.append(f"rel={fmt_ratio(s.get('reliability'))}")
            if s.get("availability_window_sec") is not None:
                extras.append(f"avail={fmt_ratio(s.get('availability_window_sec'))}s")
            if extras:
                base += f" [{' | '.join(extras)}]"
            if s.get("fallbacks"):
                fallbacks = s.get("fallbacks") or []
                formatted_fb: List[str] = []
                for entry in fallbacks:
                    if isinstance(entry, dict):
                        node = entry.get("node") or "?"
                        rel = entry.get("reliability")
                        score = entry.get("score")
                        token = node
                        if rel is not None:
                            token += f"(rel={float(rel):.2f})"
                        if score is not None:
                            token += f"@{float(score):.1f}"
                        formatted_fb.append(token)
                    else:
                        formatted_fb.append(str(entry))
                base += f" fallback→{', '.join(formatted_fb)}"
            stages_lines.append(base)
        stages_str = "\n".join(stages_lines)
        spread = r.get("federation_spread")
        resilience = r.get("resilience_score")
        cross = r.get("cross_federation_fallback_ratio")
        tbl.add_row(
            str(r.get('job_id') or '—'),
            f"{r.get('latency_ms')}",
            f"{r.get('energy_kj')}",
            f"{r.get('risk')}",
            fmt_ratio(r.get("avg_reliability")),
            fmt_ratio(spread),
            fmt_pct(resilience),
            fmt_pct(cross),
            'ok' if not r.get('infeasible') else 'not ok',
            stages_str or '—',
        )
    console.print(tbl)  # type: ignore


def plan_remote(
    base_url: str,
    jobs: List[Dict[str, Any]],
    dry_run: bool,
    strategy: Optional[str] = None,
) -> List[Dict[str, Any]]:
    import requests  # only needed in remote mode
    base = base_url.rstrip("/")

    if len(jobs) == 1:
        payload = {"job": jobs[0], "dry_run": dry_run}
        if strategy:
            payload["strategy"] = strategy
        r = requests.post(f"{base}/plan", json=payload, timeout=60)
        j = r.json()
        if not j.get("ok"):
            raise RuntimeError(f"remote /plan error: {j}")
        return [j["data"]]
    else:
        payload = {"jobs": jobs, "dry_run": dry_run}
        if strategy:
            payload["strategy"] = strategy
        r = requests.post(f"{base}/plan_batch", json=payload, timeout=120)
        j = r.json()
        if not j.get("ok"):
            raise RuntimeError(f"remote /plan_batch error: {j}")
        # /plan_batch wraps results under data.results
        data = j["data"]
        if isinstance(data, dict) and "results" in data:
            return list(data["results"])
        # fallback
        return list(data)

def plan_local(jobs: List[Dict[str, Any]], dry_run: bool, strategy: str, repeat: int) -> List[Dict[str, Any]]:
    (
        DTState,
        CostModel,
        GreedyPlanner,
        FederatedPlanner,
        MarkovPlanner,
        RLPolicy,
        BanditPolicy,
    ) = _try_import_local()
    if DTState is None:
        raise RuntimeError("Local DT modules not importable. Did you run from project root and install requirements?")

    state = DTState()
    cm = CostModel(state)
    bandit = BanditPolicy(persist_path="sim/bandit_state.json") if BanditPolicy else None
    normalized = strategy.lower()
    if normalized in {"greedy", "cheapest-energy"}:
        planner = GreedyPlanner(
            state,
            cm,
            bandit=bandit,
            cfg={
                "risk_weight": 10.0,
                "energy_weight": 0.0 if normalized == "greedy" else 0.1,
                "prefer_locality_bonus_ms": 0.5,
                "require_format_match": False,
            },
        )

        def run(job: Dict[str, Any]) -> Dict[str, Any]:
            return planner.plan_job(job, dry_run=dry_run)

    elif normalized in {"rl", "mdp", "rl-markov", "markov", "mdp-rl", "reinforcement"}:
        if MarkovPlanner is None:
            raise RuntimeError("Markov planner unavailable; ensure dt.policy.mdp is importable")
        rl_agent = RLPolicy(persist_path=os.environ.get("FABRIC_RL_STATE", "sim/rl_state.json")) if RLPolicy else None
        planner_rl = MarkovPlanner(state, cm, rl_policy=rl_agent, redundancy=3)

        def run(job: Dict[str, Any]) -> Dict[str, Any]:
            return planner_rl.plan_job(job, dry_run=dry_run)
    elif normalized in {"bandit", "bandit-greedy", "bandit-latency", "bandit-format"}:
        if BanditPolicy is None:
            raise RuntimeError("Bandit policy unavailable; ensure dt.policy.bandit is importable")
        bandit_local = BanditPolicy(persist_path=None)
        planner_bandit = GreedyPlanner(
            state,
            cm,
            bandit=bandit_local,
            cfg={
                "risk_weight": 10.0,
                "energy_weight": 0.0,
                "prefer_locality_bonus_ms": 0.5,
                "require_format_match": False,
            },
        )

        def run(job: Dict[str, Any]) -> Dict[str, Any]:
            return planner_bandit.plan_job(job, dry_run=dry_run)
    else:
        if FederatedPlanner is None:
            raise RuntimeError("Federated planner unavailable; ensure dt.policy.resilient is importable")
        planner_ft = FederatedPlanner(state, cm)

        def run(job: Dict[str, Any]) -> Dict[str, Any]:
            return planner_ft.plan_job(job, dry_run=dry_run, mode=normalized)

    results: List[Dict[str, Any]] = []
    for job in jobs:
        last = None
        times = max(1, int(repeat))
        for _ in range(times):
            res = run(job)
            # store last; bandit (if any) will learn across repeats
            last = res
        if last is not None:
            results.append(last)
    return results

def main():
    ap = argparse.ArgumentParser(description="Fabric DT — run planner over job YAML(s)")
    ap.add_argument("--job", required=True, help="Path to job YAML (single job, list of jobs, or {jobs: [...]})")
    ap.add_argument("--dry-run", action="store_true", help="Plan without reserving")
    ap.add_argument("--remote", default=None, help="Base URL of dt/api (e.g., http://127.0.0.1:8080)")
    ap.add_argument(
        "--strategy",
        default="greedy",
        choices=[
            "greedy",
            "cheapest-energy",
            "bandit",
            "resilient",
            "network-aware",
            "federated",
            "balanced",
            "fault-tolerant",
            "rl",
            "rl-markov",
            "mdp",
        ],
        help="Planner strategy to apply",
    )
    ap.add_argument("--repeat", type=int, default=1, help="Local-only: repeat planning N times (bandit learning)")
    ap.add_argument("--out", default=None, help="Write JSON results to this path")
    args = ap.parse_args()

    job_path = Path(args.job)
    if not job_path.exists():
        print(f"error: job file not found: {job_path}", file=sys.stderr)
        sys.exit(2)

    try:
        obj = load_yaml(job_path)
        jobs = ensure_jobs(obj)
    except Exception as e:
        print(f"error: failed to load/parse job YAML: {e}", file=sys.stderr)
        sys.exit(2)

    try:
        if args.remote:
            results = plan_remote(args.remote, jobs, dry_run=bool(args.dry_run), strategy=args.strategy)
        else:
            results = plan_local(jobs, dry_run=bool(args.dry_run), strategy=args.strategy, repeat=args.repeat)
    except Exception as e:
        print(f"error: planning failed: {e}", file=sys.stderr)
        sys.exit(1)

    print_summary(results)

    if args.out:
        outp = Path(args.out)
        try:
            outp.parent.mkdir(parents=True, exist_ok=True)
            outp.write_text(json.dumps(results, indent=2), encoding="utf-8")
            if RICH:
                console.print(f"[green]Saved results →[/green] {outp}")  # type: ignore
            else:
                print(f"Saved results -> {outp}")
        except Exception as e:
            print(f"warn: failed to write --out file: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()

