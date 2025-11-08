#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dt/api.py — Flask API for the Fabric Digital Twin

Endpoints
---------
GET  /health
GET  /snapshot
POST /observe            { payload: {type: "node"|"link", ...} }
POST /plan               { job: {...}, dry_run?: bool, strategy?: "greedy"|"cheapest-energy" }
POST /plan_batch         { jobs: [ {...}, ... ], dry_run?: bool, strategy?: ... }
POST /release            { releases: [ {node: "...", reservation_id: "..."} ] }

Run
---
export FLASK_APP=dt.api:app
flask run -h 0.0.0.0 -p 8080

or:

python3 -m dt.api --host 0.0.0.0 --port 8080
"""

from __future__ import annotations
import argparse
import json
import os
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional
import logging
from flask import Flask, jsonify, request

from .state import DTState, safe_float, safe_int
from .cost_model import CostModel
from .exporters import as_dtdl, as_k8s_crds
from .policy.resilient import FederatedPlanner
from .policy.mdp import MarkovPlanner
from .policy.rl_stub import RLPolicy
from .policy.greedy import GreedyPlanner

try:
    from .policy.bandit import BanditPolicy
except Exception:  # pragma: no cover
    BanditPolicy = None  # type: ignore

import yaml

# -----------------------------------
# App singletons
# -----------------------------------

STATE = DTState()  # loads nodes/, topology, starts watcher
CM = CostModel(STATE)
FED_PLANNER = FederatedPlanner(STATE, CM)
RL_AGENT = RLPolicy(persist_path=os.environ.get("FABRIC_RL_STATE", "sim/rl_state.json"))
MDP_PLANNER = MarkovPlanner(
    STATE,
    CM,
    rl_policy=RL_AGENT,
    gamma=0.92,
    failure_penalty=10.0,
    redundancy=3,
)
BANDIT_POLICY = (
    BanditPolicy(persist_path=os.environ.get("FABRIC_BANDIT_STATE", "sim/bandit_state.json"))
    if BanditPolicy
    else None
)
GREEDY_LATENCY = GreedyPlanner(
    STATE,
    CM,
    bandit=None,
    cfg={
        "risk_weight": 10.0,
        "energy_weight": 0.0,
        "prefer_locality_bonus_ms": 0.5,
        "require_format_match": False,
    },
)
GREEDY_ENERGY = GreedyPlanner(
    STATE,
    CM,
    bandit=None,
    cfg={
        "risk_weight": 10.0,
        "energy_weight": 0.1,
        "prefer_locality_bonus_ms": 0.5,
        "require_format_match": False,
    },
)
GREEDY_BANDIT = GreedyPlanner(
    STATE,
    CM,
    bandit=BANDIT_POLICY,
    cfg={
        "risk_weight": 10.0,
        "energy_weight": 0.0,
        "prefer_locality_bonus_ms": 0.5,
        "require_format_match": False,
    },
)

RECENT_PLANS: Deque[Dict[str, Any]] = deque(maxlen=200)

app = Flask(__name__)


# -----------------------------------
# Helpers
# -----------------------------------


def _ok(data: Any, status: int = 200):
    return jsonify({"ok": True, "data": data}), status


def _err(msg: str, status: int = 400, **extra):
    return jsonify({"ok": False, "error": msg, **extra}), status


def _jobs_root() -> Path:
    base = os.environ.get("FABRIC_JOBS_ROOT", "jobs")
    return Path(base).resolve()


def _load_job_catalog() -> List[Dict[str, Any]]:
    root = _jobs_root()
    entries: List[Dict[str, Any]] = []
    if not root.exists():
        return entries
    for path in sorted(root.glob("*.y*ml")):
        try:
            content = yaml.safe_load(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        jobs = _ensure_jobs(content)
        for idx, job in enumerate(jobs):
            job_id = job.get("id") or f"{path.name}#{idx + 1}"
            entries.append(
                {
                    "id": job_id,
                    "file": path.name,
                    "index": idx,
                    "path": str(path),
                    "job": job,
                }
            )
    return entries


def _ensure_jobs(obj: Any) -> List[Dict[str, Any]]:
    if isinstance(obj, list):
        return [item for item in obj if isinstance(item, dict)]
    if isinstance(obj, dict):
        if isinstance(obj.get("jobs"), list):
            return [item for item in obj["jobs"] if isinstance(item, dict)]
        return [obj]
    return []


# -----------------------------------
# Routes
# -----------------------------------


@app.get("/health")
def health():
    return _ok({"ts": STATE.snapshot()["ts"]})


@app.get("/snapshot")
def snapshot():
    return _ok(STATE.snapshot())


@app.post("/observe")
def observe():
    if not request.is_json:
        return _err("expected JSON body")
    try:
        payload = request.get_json()
        STATE.apply_observation(payload)
        return _ok({"applied": True})
    except Exception as e:
        logging.exception("Exception in /observe endpoint")
        return _err("observe failed: internal error")


@app.post("/plan")
def plan():
    """
    Plan a single job.
    Body:
    {
      "job": { id, deadline_ms?, stages:[ {id, size_mb?, resources{cpu_cores,mem_gb,gpu_vram_gb}, allowed_formats?, ...}, ...] },
      "strategy": "greedy"|"cheapest-energy",
      "dry_run": false
    }
    """
    if not request.is_json:
        return _err("expected JSON body")
    body = request.get_json() or {}
    job = body.get("job")
    if not job:
        return _err("missing 'job'")

    strategy_raw = body.get("strategy") or "greedy"
    strategy = strategy_raw.lower().strip()
    dry_run = bool(body.get("dry_run", False))

    if strategy in {
        "resilient",
        "network-aware",
        "federated",
        "fault-tolerant",
        "ft",
        "failover",
        "balanced",
        "load-balance",
        "load-balanced",
    }:
        planner_result = FED_PLANNER.plan_job(job, dry_run=dry_run, mode=strategy)
    elif strategy in {"rl", "mdp", "rl-markov", "markov", "mdp-rl", "reinforcement"}:
        planner_result = MDP_PLANNER.plan_job(job, dry_run=dry_run)
    else:
        if strategy in {"cheapest-energy", "energy", "energy-aware"}:
            planner_obj = GREEDY_ENERGY
        elif strategy in {"bandit", "bandit-greedy", "bandit-latency", "bandit-format"}:
            planner_obj = GREEDY_BANDIT if GREEDY_BANDIT is not None else GREEDY_LATENCY
        else:
            planner_obj = GREEDY_LATENCY
        planner_result = planner_obj.plan_job(job, dry_run=dry_run)

    ddl = safe_float(job.get("deadline_ms"), 0.0)
    penalty = CM.slo_penalty(ddl, planner_result.get("latency_ms", 0.0)) if ddl > 0 else 0.0

    planner_result["strategy"] = strategy_raw
    planner_result["dry_run"] = dry_run
    planner_result.setdefault("per_stage", [])
    planner_result.setdefault("reservations", [])
    planner_result.setdefault("assignments", {})
    planner_result.setdefault("federation_summary", STATE.federations_overview())
    planner_result["deadline_ms"] = ddl or None
    planner_result["slo_penalty"] = penalty
    planner_result["ts"] = int(time.time() * 1000)
    planner_result.setdefault("avg_reliability", planner_result.get("avg_reliability"))
    planner_result["predictive"] = STATE.predictive_overview()
    RECENT_PLANS.appendleft(planner_result)
    return _ok(planner_result)


@app.get("/plans")
def plans():
    return _ok(list(RECENT_PLANS))


@app.get("/jobs")
def jobs():
    return _ok(_load_job_catalog())


@app.get("/events")
def events():
    since = request.args.get("since")
    limit_int = safe_int(request.args.get("limit", 100), 100)
    limit_int = max(1, min(limit_int, 500))
    recent = STATE.recent_events(limit=limit_int, since_id=since)
    return _ok({"events": recent, "limit": limit_int, "since": since})


@app.get("/standards/dtdl")
def standards_dtdl():
    return _ok(as_dtdl(STATE))


@app.get("/standards/crds")
def standards_crds():
    return _ok(as_k8s_crds(STATE))


@app.post("/plan_batch")
def plan_batch():
    """
    Plan multiple jobs in one call.
    Body:
    {
      "jobs": [ { job }, ... ],
      "strategy": "...",
      "dry_run": false
    }
    """
    if not request.is_json:
        return _err("expected JSON body")
    body = request.get_json() or {}
    jobs = body.get("jobs") or []
    if not jobs:
        return _err("missing 'jobs'")

    strategy = (body.get("strategy") or "greedy").lower().strip()
    dry_run = bool(body.get("dry_run", False))

    results = []
    for j in jobs:
        # Reuse the logic by faking a request-local plan
        tmp_req = {"job": j, "strategy": strategy, "dry_run": dry_run}
        with app.test_request_context(json=tmp_req):
            plan_result = app.view_functions["plan"]()  # type: ignore
            if isinstance(plan_result, tuple):
                resp_obj, status = plan_result
            else:
                resp_obj = plan_result
                status = getattr(resp_obj, "status_code", 200)

            data = None
            if hasattr(resp_obj, "get_json"):
                data = resp_obj.get_json(silent=True)
            if data is None:
                try:
                    data = json.loads(resp_obj.get_data(as_text=True) or "{}")
                except Exception:
                    data = {}

            if status >= 400 or not data.get("ok", False):
                return resp_obj, status

            payload = data.get("data")
            if payload is None:
                return _err("plan returned no data", status=500)

            results.append(payload)
    return _ok({"results": results})


@app.post("/release")
def release():
    """
    Release reservations.
    Body: { releases: [ { node: "ws-001", reservation_id: "res-0000001" }, ... ] }
    """
    if not request.is_json:
        return _err("expected JSON body")
    body = request.get_json() or {}
    rels = body.get("releases") or []
    done = []
    for r in rels:
        node = r.get("node")
        rid = r.get("reservation_id")
        if node and rid:
            ok = STATE.release(node, rid)
            done.append({"node": node, "reservation_id": rid, "released": bool(ok)})
    return _ok({"released": done})


# -----------------------------------
# CLI entrypoint
# -----------------------------------


def main():
    ap = argparse.ArgumentParser(description="Fabric DT API")
    ap.add_argument("--host", default=os.environ.get("FABRIC_API_HOST", "127.0.0.1"))
    ap.add_argument(
        "--port", type=int, default=int(os.environ.get("FABRIC_API_PORT", "8080"))
    )
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
