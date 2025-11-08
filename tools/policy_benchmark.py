#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate comparison plots for DT planner policies."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
import warnings

warnings.filterwarnings(
    "ignore", message="invalid value encountered in dot", category=RuntimeWarning
)

from dt.cost_model import CostModel
from dt.policy.greedy import GreedyPlanner
from dt.policy.mdp import MarkovPlanner
from dt.policy.resilient import FederatedPlanner
from dt.policy.rl_stub import RLPolicy
from dt.state import DTState

try:
    from dt.policy.bandit import BanditPolicy
except Exception:  # pragma: no cover
    BanditPolicy = None  # type: ignore


STRATEGY_LABELS: Dict[str, str] = {
    "greedy": "Greedy",
    "cheapest-energy": "Cheapest energy",
    "bandit": "Bandit",
    "rl": "RL Markov",
    "rl-markov": "RL Markov",
    "mdp": "RL Markov",
    "resilient": "Resilient",
    "network-aware": "Network aware",
    "federated": "Federated",
    "balanced": "Balanced",
}


def load_jobs(paths: Sequence[Path]) -> List[Dict[str, object]]:
    jobs: List[Dict[str, object]] = []
    for path in paths:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            jobs.extend([j for j in data if isinstance(j, dict)])
        elif isinstance(data, dict):
            if isinstance(data.get("jobs"), list):
                jobs.extend([j for j in data["jobs"] if isinstance(j, dict)])
            else:
                jobs.append(data)
        else:
            raise ValueError(f"Unsupported YAML structure in {path}")
    if not jobs:
        raise ValueError("No jobs discovered from the provided YAML paths")
    return jobs


def _greedy_planner(state: DTState, cm: CostModel, *, bandit=False, energy=False) -> GreedyPlanner:
    cfg = {
        "risk_weight": 10.0,
        "energy_weight": 0.1 if energy else 0.0,
        "prefer_locality_bonus_ms": 0.5,
        "require_format_match": False,
    }
    policy = None
    if bandit:
        if BanditPolicy is None:
            raise RuntimeError("BanditPolicy not available; install dt.policy.bandit dependencies")
        policy = BanditPolicy(persist_path=None)
    return GreedyPlanner(state, cm, bandit=policy, cfg=cfg)


def evaluate_strategy(
    strategy: str,
    jobs: Iterable[Dict[str, object]],
    state: DTState,
    cm: CostModel,
) -> List[Dict[str, object]]:
    norm = strategy.lower().strip()
    if norm in {"greedy"}:
        planner = _greedy_planner(state, cm, bandit=False, energy=False)
        return [planner.plan_job(job, dry_run=True) for job in jobs]
    if norm in {"cheapest-energy", "energy"}:
        planner = _greedy_planner(state, cm, bandit=False, energy=True)
        return [planner.plan_job(job, dry_run=True) for job in jobs]
    if norm in {"bandit", "bandit-greedy", "bandit-latency", "bandit-format"}:
        planner = _greedy_planner(state, cm, bandit=True, energy=False)
        return [planner.plan_job(job, dry_run=True) for job in jobs]
    if norm in {"rl", "rl-markov", "mdp", "markov", "reinforcement"}:
        rl_agent = RLPolicy(persist_path=None)
        planner = MarkovPlanner(state, cm, rl_policy=rl_agent, redundancy=3)
        return [planner.plan_job(job, dry_run=True) for job in jobs]
    planner = FederatedPlanner(state, cm)
    return [planner.plan_job(job, dry_run=True, mode=norm) for job in jobs]


def aggregate_metrics(results: List[Dict[str, object]]) -> Dict[str, float]:
    latency = []
    energy = []
    risk = []
    infeasible = 0
    for item in results:
        try:
            latency.append(float(item.get("latency_ms", 0.0)))
        except Exception:
            pass
        try:
            energy.append(float(item.get("energy_kj", 0.0)))
        except Exception:
            pass
        try:
            risk.append(float(item.get("risk", 0.0)))
        except Exception:
            pass
        if item.get("infeasible"):
            infeasible += 1
    n = max(1, len(results))
    return {
        "latency_avg": float(np.mean(latency)) if latency else 0.0,
        "energy_avg": float(np.mean(energy)) if energy else 0.0,
        "risk_avg": float(np.mean(risk)) if risk else 0.0,
        "infeasible_ratio": infeasible / n,
    }


def plot_metrics(summary: Dict[str, Dict[str, float]], out_path: Path) -> None:
    strategies = list(summary.keys())
    idx = np.arange(len(strategies))
    latency = [summary[s]["latency_avg"] for s in strategies]
    energy = [summary[s]["energy_avg"] for s in strategies]
    risk = [summary[s]["risk_avg"] for s in strategies]
    infeasible = [summary[s]["infeasible_ratio"] * 100.0 for s in strategies]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    labels = [STRATEGY_LABELS.get(s, s) for s in strategies]

    axes[0].bar(idx, latency, color="#4c72b0")
    axes[0].set_title("Latency (ms)")
    axes[0].set_xticks(idx)
    axes[0].set_xticklabels(labels)
    axes[0].tick_params(axis="x", labelrotation=20)

    axes[1].bar(idx, energy, color="#55a868")
    axes[1].set_title("Energy (kJ)")
    axes[1].set_xticks(idx)
    axes[1].tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)

    axes[2].bar(idx, risk, color="#c44e52")
    axes[2].set_title("Risk score")
    axes[2].set_xticks(idx)
    axes[2].tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)

    axes[3].bar(idx, infeasible, color="#8172b3")
    axes[3].set_title("Infeasible (%)")
    axes[3].set_xticks(idx)
    axes[3].tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)

    for ax in axes:
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    fig.suptitle("Fabric DT policy comparison", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark DT planner policies and plot metrics")
    parser.add_argument(
        "--jobs",
        nargs="+",
        required=True,
        help="Paths to job YAML files (single job or list)",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["greedy", "bandit", "rl-markov"],
        help="Planner strategies to evaluate",
    )
    parser.add_argument(
        "--out",
        default="reports/policy_metrics.png",
        help="Path to save the generated plot",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=6,
        help="Maximum number of jobs to sample (per run) for fast comparisons",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="Optional path to write aggregated metrics as JSON",
    )
    args = parser.parse_args()

    job_paths = [Path(p) for p in args.jobs]
    jobs = load_jobs(job_paths)
    limit = max(1, int(args.limit))
    if len(jobs) > limit:
        jobs = jobs[:limit]

    state = DTState()
    cm = CostModel(state)

    summary: Dict[str, Dict[str, float]] = {}
    detailed: Dict[str, List[Dict[str, object]]] = {}
    for strategy in args.strategies:
        results = evaluate_strategy(strategy, jobs, state, cm)
        summary[strategy] = aggregate_metrics(results)
        detailed[strategy] = results

    out_path = Path(args.out)
    plot_metrics(summary, out_path)

    if args.json_out:
        json_path = Path(args.json_out)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps({"summary": summary, "results": detailed}, indent=2), encoding="utf-8")

    print("Saved plot →", out_path)
    if args.json_out:
        print("Saved metrics →", args.json_out)


if __name__ == "__main__":
    main()
