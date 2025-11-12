#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ui/dashboard.py — Fabric DT web dashboard (full, single file).

Features
--------
- Overview: totals, quick stats
- Nodes table: status, arch, formats, effective/free capacities, health flags
- Links table: speed/RTT/jitter/loss/ECN
- Recent plans: per-stage node, latency/energy/risk, strategy, dry_run
- Actions:
  • Submit job JSON for planning (dry run / with reservations)
  • Run demo job
  • Apply observation (node/link changes)
  • Refresh snapshot

Run
---
python3 -m ui.dashboard --host 0.0.0.0 --port 8090
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from flask import Flask, jsonify, request, make_response

# --- DT imports ---
from dt.state import DTState, safe_float
from dt.cost_model import CostModel
from dt.policy.greedy import GreedyPlanner
from dt.policy.resilient import FederatedPlanner
from dt.policy.mdp import MarkovPlanner
from dt.policy.rl_stub import RLPolicy

try:
    from dt.policy.bandit import BanditPolicy
except Exception:
    BanditPolicy = None  # optional

import yaml

from sim.chaos import (
    ChaosEngine,
    OverridesStore,
    collect_chaos_events,
    load_nodes_index,
    load_topology,
)

# ----------------- App singletons -----------------

app = Flask(__name__)

REMOTE_BASE = os.environ.get("FABRIC_DT_REMOTE")
REMOTE_TIMEOUT = float(os.environ.get("FABRIC_DT_REMOTE_TIMEOUT", "10.0"))
REMOTE_LABEL = "local DT (embedded)"
SESSION = requests.Session()

STATE: Optional[DTState] = None
CM: Optional[CostModel] = None
BANDIT = None
PLANNER_GREEDY: Optional[GreedyPlanner] = None
PLANNER_ENERGY: Optional[GreedyPlanner] = None
PLANNER_BANDIT: Optional[GreedyPlanner] = None
FED_PLANNER: Optional[FederatedPlanner] = None
RL_AGENT: Optional[RLPolicy] = None
MDP_PLANNER: Optional[MarkovPlanner] = None

RECENT_PLANS: deque = deque(maxlen=50)

CHAOS_THREAD: Optional[threading.Thread] = None
CHAOS_ENGINE: Optional[ChaosEngine] = None
CHAOS_STATUS: Dict[str, Any] = {
    "running": False,
    "scenario": None,
    "speed": 1.0,
    "events": 0,
    "last_error": None,
}
CHAOS_LOCK = threading.Lock()


def _configure_runtime(remote: Optional[str], timeout: float) -> None:
    global STATE, CM, BANDIT, PLANNER_GREEDY, PLANNER_ENERGY, PLANNER_BANDIT, FED_PLANNER, RL_AGENT, MDP_PLANNER, REMOTE_BASE, REMOTE_TIMEOUT, REMOTE_LABEL

    REMOTE_BASE = remote.rstrip("/") if remote else None
    REMOTE_TIMEOUT = timeout
    REMOTE_LABEL = (
        f"remote API @ {REMOTE_BASE}" if REMOTE_BASE else "local DT (embedded)"
    )

    if REMOTE_BASE:
        STATE = None
        CM = None
        BANDIT = None
        PLANNER_GREEDY = None
        PLANNER_ENERGY = None
        PLANNER_BANDIT = None
        FED_PLANNER = None
        RL_AGENT = None
        MDP_PLANNER = None
    else:
        if STATE is None:
            STATE = DTState()
            CM = CostModel(STATE)
            BANDIT = (
                BanditPolicy(persist_path="sim/bandit_state.json")
                if BanditPolicy
                else None
            )
        if STATE is not None and CM is not None:
            PLANNER_GREEDY = GreedyPlanner(
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
            PLANNER_ENERGY = GreedyPlanner(
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
            PLANNER_BANDIT = GreedyPlanner(
                STATE,
                CM,
                bandit=BANDIT,
                cfg={
                    "risk_weight": 10.0,
                    "energy_weight": 0.0,
                    "prefer_locality_bonus_ms": 0.5,
                    "require_format_match": False,
                },
            )
            FED_PLANNER = FederatedPlanner(STATE, CM)
            RL_AGENT = RLPolicy(persist_path=os.environ.get("FABRIC_RL_STATE", "sim/rl_state.json"))
            MDP_PLANNER = MarkovPlanner(
                STATE,
                CM,
                rl_policy=RL_AGENT,
                redundancy=3,
            )


_configure_runtime(REMOTE_BASE, REMOTE_TIMEOUT)


# ----------------- Helpers -----------------


def _ok(data: Any, status: int = 200):
    return jsonify({"ok": True, "data": data}), status


def _err(msg: str, status: int = 400, **extra):
    return jsonify({"ok": False, "error": msg, **extra}), status


def _remote_url(path: str) -> str:
    if not REMOTE_BASE:
        raise RuntimeError("remote base not configured")
    if not path.startswith("/"):
        path = "/" + path
    return f"{REMOTE_BASE}{path}"


class DTOverridesStore(OverridesStore):
    """Overrides store that also mutates the embedded DT state."""

    def __init__(self, path: Path, state: Optional[DTState]):
        super().__init__(path)
        self._state = state

    def _apply_to_state(self, payload: Dict[str, Any], action: str) -> None:
        if not self._state:
            return
        typ = payload.get("type")
        if typ == "node":
            node = payload.get("node")
            if not node:
                return
            if action == "apply":
                changes = dict(payload.get("changes") or {})
            else:
                fields = list(payload.get("fields") or [])
                changes = {field: None for field in fields}
            if not changes:
                return
            self._state.apply_observation(
                {"payload": {"type": "node", "node": node, "changes": changes}}
            )
        elif typ == "link":
            key = payload.get("key")
            if not key:
                return
            if action == "apply":
                changes = dict(payload.get("changes") or {})
            else:
                fields = list(payload.get("fields") or [])
                changes = {field: None for field in fields}
            if not changes:
                return
            self._state.apply_observation(
                {"payload": {"type": "link", "key": key, "changes": changes}}
            )

    @staticmethod
    def _link_key(a: str, b: str) -> str:
        return "|".join(sorted([a, b]))

    def link_apply(self, a: str, b: str, changes: Dict[str, Any]):
        key = self._link_key(a, b)
        super().link_apply(a, b, changes)
        self._apply_to_state({"type": "link", "key": key, "changes": dict(changes or {})}, "apply")

    def link_revert(self, a: str, b: str, fields: List[str]):
        key = self._link_key(a, b)
        super().link_revert(a, b, fields)
        self._apply_to_state({"type": "link", "key": key, "fields": list(fields or [])}, "revert")

    def node_apply(self, node: str, changes: Dict[str, Any]):
        super().node_apply(node, changes)
        self._apply_to_state({"type": "node", "node": node, "changes": dict(changes or {})}, "apply")

    def node_revert(self, node: str, fields: List[str]):
        super().node_revert(node, fields)
        self._apply_to_state({"type": "node", "node": node, "fields": list(fields or [])}, "revert")


def _local_state_available() -> bool:
    return STATE is not None and not REMOTE_BASE


def _chaos_paths() -> Dict[str, Path]:
    if not _local_state_available():
        raise RuntimeError("embedded DT state unavailable")
    assert STATE is not None
    return {
        "topology": STATE.topology_path,
        "overrides": STATE.overrides_path,
        "nodes": STATE.nodes_dir,
    }


def _build_chaos_engine(scenario: Optional[str], speed: float) -> Dict[str, Any]:
    paths = _chaos_paths()
    topology = load_topology(paths["topology"])
    schedule = collect_chaos_events(topology, scenario)
    nodes_index = load_nodes_index(paths["nodes"])
    store = DTOverridesStore(paths["overrides"], STATE)
    engine = ChaosEngine(
        store,
        speed=max(0.1, speed),
        verbose=False,
        nodes_index=nodes_index,
    )
    return {"engine": engine, "schedule": schedule}


def _chaos_worker(engine: ChaosEngine, schedule: List[Any]):
    global CHAOS_THREAD, CHAOS_ENGINE
    try:
        engine.run(schedule)
    except Exception as exc:
        with CHAOS_LOCK:
            CHAOS_STATUS["last_error"] = str(exc)
    finally:
        with CHAOS_LOCK:
            CHAOS_STATUS["running"] = False
            CHAOS_ENGINE = None
            CHAOS_THREAD = None


def _chaos_status() -> Dict[str, Any]:
    with CHAOS_LOCK:
        running = bool(CHAOS_THREAD and CHAOS_THREAD.is_alive())
        status = dict(CHAOS_STATUS)
        status["running"] = running
    return status


def _update_chaos_status_running(scenario: Optional[str], speed: float, events: int):
    with CHAOS_LOCK:
        CHAOS_STATUS.update(
            {
                "running": True,
                "scenario": scenario,
                "speed": speed,
                "events": events,
                "last_error": None,
            }
        )


def _set_chaos_idle():
    with CHAOS_LOCK:
        CHAOS_STATUS.update({"running": False})


def _proxy_remote(method: str, path: str, payload: Optional[Dict[str, Any]] = None):
    if not REMOTE_BASE:
        raise RuntimeError("remote base not configured")
    try:
        kwargs: Dict[str, Any] = {"timeout": REMOTE_TIMEOUT}
        if payload is not None:
            kwargs["json"] = payload
        resp = SESSION.request(method.upper(), _remote_url(path), **kwargs)
    except requests.RequestException as exc:
        return _err(f"remote request failed: {exc}", status=502)

    try:
        data = resp.json()
    except ValueError:
        return _err(
            f"remote returned non-JSON response (status {resp.status_code})", status=502
        )

    return make_response(data, resp.status_code)


def _demo_job() -> Dict[str, Any]:
    return {
        "id": f"demo-{int(time.time())}",
        "deadline_ms": 5000,
        "stages": [
            {
                "id": "ingest",
                "type": "io",
                "size_mb": 40,
                "resources": {"cpu_cores": 1, "mem_gb": 1},
                "allowed_formats": ["native", "wasm"],
            },
            {
                "id": "prep",
                "type": "preproc",
                "size_mb": 60,
                "resources": {"cpu_cores": 2, "mem_gb": 2},
                "allowed_formats": ["native", "cuda", "wasm"],
            },
            {
                "id": "mlp",
                "type": "mlp",
                "size_mb": 100,
                "resources": {"cpu_cores": 4, "mem_gb": 4, "gpu_vram_gb": 2},
                "allowed_formats": ["cuda", "native"],
            },
        ],
    }


# ----------------- JSON APIs -----------------


def _plan_local_job(job: Dict[str, Any], dry: bool, strategy: str) -> Dict[str, Any]:
    normalized = strategy.lower().strip()
    if normalized in {"resilient", "network-aware", "federated", "balanced", "fault-tolerant"}:
        if FED_PLANNER is None:
            raise RuntimeError("Federated planner unavailable in embedded mode")
        res = FED_PLANNER.plan_job(job, dry_run=dry, mode=normalized)
    elif normalized in {"rl", "mdp", "rl-markov", "markov", "mdp-rl", "reinforcement"}:
        if MDP_PLANNER is None:
            raise RuntimeError("RL Markov planner unavailable in embedded mode")
        res = MDP_PLANNER.plan_job(job, dry_run=dry)
    else:
        planner = PLANNER_GREEDY
        if normalized in {"cheapest-energy", "energy", "energy-aware"} and PLANNER_ENERGY is not None:
            planner = PLANNER_ENERGY
        elif normalized in {"bandit", "bandit-greedy", "bandit-latency", "bandit-format"} and PLANNER_BANDIT is not None:
            planner = PLANNER_BANDIT
        if planner is None:
            raise RuntimeError("Greedy planner unavailable in embedded mode")
        res = planner.plan_job(job, dry_run=dry)

    ddl = safe_float(job.get("deadline_ms"), 0.0)
    if ddl > 0 and CM is not None:
        res["slo_penalty"] = CM.slo_penalty(ddl, res.get("latency_ms", 0.0))
        res["deadline_ms"] = ddl
    res["strategy"] = strategy
    res["dry_run"] = dry
    res["ts"] = int(time.time() * 1000)
    return res


def _jobs_root() -> Path:
    base = os.environ.get("FABRIC_JOBS_ROOT", "jobs")
    return Path(base).resolve()


def _ensure_jobs(obj: Any) -> List[Dict[str, Any]]:
    if isinstance(obj, list):
        return [item for item in obj if isinstance(item, dict)]
    if isinstance(obj, dict):
        if isinstance(obj.get("jobs"), list):
            return [item for item in obj["jobs"] if isinstance(item, dict)]
        return [obj]
    return []


def _load_job_catalog() -> List[Dict[str, Any]]:
    root = _jobs_root()
    entries: List[Dict[str, Any]] = []
    if not root.exists():
        return entries
    for path in sorted(root.glob("*.y*ml")):
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        jobs = _ensure_jobs(payload)
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


def _load_chaos_scenarios() -> Dict[str, Any]:
    if not _local_state_available():
        return {"scenarios": [], "has_base": False}
    try:
        paths = _chaos_paths()
        topology = load_topology(paths["topology"])
    except Exception as exc:
        raise RuntimeError(f"failed to load topology: {exc}") from exc
    scenarios: List[Dict[str, Any]] = []
    for sc in (topology.get("scenarios") or []):
        if not isinstance(sc, dict):
            continue
        name = sc.get("name")
        if not name:
            continue
        scenarios.append({
            "name": name,
            "description": sc.get("description"),
        })
    has_base = bool(topology.get("chaos"))
    return {"scenarios": scenarios, "has_base": has_base}


@app.get("/api/chaos_scenarios")
def api_chaos_scenarios():
    if REMOTE_BASE:
        return _ok({"scenarios": [], "has_base": False, "remote": True})
    try:
        data = _load_chaos_scenarios()
    except Exception as exc:
        return _err(str(exc))
    return _ok(data)


@app.get("/api/chaos")
def api_chaos_status():
    if REMOTE_BASE:
        return _ok({"running": False, "remote": True})
    return _ok(_chaos_status())


@app.post("/api/chaos")
def api_chaos_control():
    global CHAOS_ENGINE, CHAOS_THREAD
    if REMOTE_BASE:
        return _err("Chaos controls unavailable when using a remote DT")
    if not request.is_json:
        return _err("expected JSON body")
    payload = request.get_json() or {}
    action = (payload.get("action") or "start").strip().lower()
    if action == "stop":
        with CHAOS_LOCK:
            engine = CHAOS_ENGINE
            thread = CHAOS_THREAD
        if not engine or not thread or not thread.is_alive():
            return _err("Chaos runner is not active")
        engine.stop()
        thread.join(timeout=2.0)
        return _ok(_chaos_status())

    scenario = payload.get("scenario")
    speed = safe_float(payload.get("speed"), 12.0)
    if speed <= 0:
        speed = 1.0
    try:
        bundle = _build_chaos_engine(scenario, speed)
    except Exception as exc:
        return _err(str(exc))
    schedule = bundle.get("schedule") or []
    if not schedule:
        return _err("Selected scenario has no chaos events")
    with CHAOS_LOCK:
        if CHAOS_THREAD and CHAOS_THREAD.is_alive():
            return _err("Chaos runner is already active")
        engine = bundle["engine"]
        thread = threading.Thread(
            target=_chaos_worker,
            args=(engine, schedule),
            name="ChaosRunner",
            daemon=True,
        )
        CHAOS_ENGINE = engine
        CHAOS_THREAD = thread
        _update_chaos_status_running(scenario, speed, len(schedule))
        thread.start()
    return _ok(_chaos_status())


@app.get("/api/health")
def api_health():
    if REMOTE_BASE:
        return _proxy_remote("GET", "/health")
    return _ok({"ts": STATE.snapshot()["ts"]})


@app.get("/api/snapshot")
def api_snapshot():
    if REMOTE_BASE:
        return _proxy_remote("GET", "/snapshot")
    return _ok(STATE.snapshot())


@app.get("/api/plans")
def api_plans():
    if REMOTE_BASE:
        return _proxy_remote("GET", "/plans")
    return _ok(list(RECENT_PLANS))


@app.get("/api/jobs")
def api_jobs():
    if REMOTE_BASE:
        return _proxy_remote("GET", "/jobs")
    return _ok(_load_job_catalog())


@app.post("/api/plan")
def api_plan():
    """
    Body:
    {
      "job": { ... }   # job JSON
      "dry_run": true|false,
      "strategy": "greedy"  # (ignored here; GreedyPlanner used)
    }
    """
    if not request.is_json:
        return _err("expected JSON body")
    body = request.get_json() or {}
    job = body.get("job")
    if not job:
        return _err("missing 'job'")
    dry = bool(body.get("dry_run", True))
    strategy = body.get("strategy") or "greedy"
    normalized = strategy.lower().strip()
    if REMOTE_BASE:
        payload = {
            "job": job,
            "dry_run": dry,
            "strategy": strategy,
        }
        return _proxy_remote("POST", "/plan", payload)

    try:
        res = _plan_local_job(job, dry, strategy)
    except Exception as exc:
        return _err(f"plan failed: {exc}")
    RECENT_PLANS.appendleft(res)
    return _ok(res)


@app.post("/api/plan_demo")
def api_plan_demo():
    job = _demo_job()
    payload_in = request.get_json() if request.is_json else {}
    dry = bool((payload_in or {}).get("dry_run", True))
    strategy = (payload_in or {}).get("strategy") or "greedy"
    normalized = strategy.lower().strip()
    if REMOTE_BASE:
        payload = {"job": job, "dry_run": dry, "strategy": strategy}
        return _proxy_remote("POST", "/plan", payload)

    try:
        res = _plan_local_job(job, dry, strategy)
    except Exception as exc:
        return _err(f"plan failed: {exc}")
    RECENT_PLANS.appendleft(res)
    return _ok(res)


@app.post("/api/plan_batch")
def api_plan_batch():
    if not request.is_json:
        return _err("expected JSON body")
    payload = request.get_json() or {}
    jobs = payload.get("jobs") or []
    if not isinstance(jobs, list) or not jobs:
        return _err("jobs must be a non-empty list")
    dry = bool(payload.get("dry_run", True))
    strategy = payload.get("strategy") or "greedy"
    if REMOTE_BASE:
        proxy_payload = {"jobs": jobs, "dry_run": dry, "strategy": strategy}
        return _proxy_remote("POST", "/plan_batch", proxy_payload)

    results: List[Dict[str, Any]] = []
    for job in jobs:
        if not isinstance(job, dict):
            continue
        try:
            res = _plan_local_job(job, dry, strategy)
        except Exception as exc:
            return _err(f"plan failed: {exc}")
        results.append(res)
        RECENT_PLANS.appendleft(res)
    return _ok({"results": results})


@app.post("/api/observe")
def api_observe():
    """
    Body: same shape as dt/state.apply_observation payload:
    { "payload": { "type":"node", "node":"name", "changes": { "down": true, "thermal_derate": 0.3 } } }
    or
    { "payload": { "type":"link", "key":"a|b", "changes": { "loss_pct": 2.0 } } }
    """
    if not request.is_json:
        return _err("expected JSON body")
    try:
        payload = request.get_json()
        if REMOTE_BASE:
            return _proxy_remote("POST", "/observe", payload)

        STATE.apply_observation(payload)
        # Optionally persist overrides so state watcher picks them up across restarts
        try:
            STATE.write_overrides()
        except Exception:
            pass
        return _ok({"applied": True})
    except Exception as e:
        return _err(f"observe failed: {e}")


# ----------------- HTML UI -----------------

_INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Fabric DT Dashboard</title>
<meta name="viewport" content="width=device-width, initial-scale=1" />
<link rel="icon" href="data:,">
<style>
:root {
  --bg: #0b0f14;
  --panel: #121822;
  --muted: #9bb0c9;
  --muted2: #6c7a8a;
  --text: #e7eef7;
  --accent: #6fc1ff;
  --accent-strong: #4aa3f0;
  --good: #2ecc71;
  --warn: #f1c40f;
  --bad: #e74c3c;
  --chip: #1a2330;
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--bg); color: var(--text);
  font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial;
}
header {
  padding: 16px 20px; background: linear-gradient(180deg, #0f1722 0%, #0b0f14 100%);
  border-bottom: 1px solid #1c2430;
  position: sticky; top: 0; z-index: 10;
}
h1 { margin: 0; font-size: 20px; letter-spacing: 0.5px; }
small { color: var(--muted2); }
.container { padding: 16px 20px; display: grid; grid-template-columns: 1fr; gap: 16px; }

.grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
@media (max-width: 1080px) { .grid-2 { grid-template-columns: 1fr; } }

.card {
  background: var(--panel); border: 1px solid #1a2533; border-radius: 12px;
  padding: 14px; box-shadow: 0 6px 20px rgba(0,0,0,0.25);
}
.card h2 { margin: 0 0 10px 0; font-size: 16px; color: #cfe7ff; letter-spacing: .3px; }
.row { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }
.kpi { background: var(--chip); padding: 10px 12px; border-radius: 10px; border: 1px solid #243140; }
.kpi .v { font-weight: 700; }
.btn {
  appearance: none; border: 1px solid #2a3a4f; background: #192434; color: var(--text);
  padding: 8px 12px; border-radius: 8px; cursor: pointer; font-weight: 600;
}
.btn:hover { border-color: #3f5876; }
.btn.primary { background: #13314d; border-color: #2c5b86; color: #cfe7ff; }
.btn.secondary { background: #172235; border-color: #2d3f57; color: #b6cae4; }
.btn.bad { background: #3a1010; border-color: #5b1a1a; color: #ffbbbb; }
.btn.good { background: #103a28; border-color: #1b5e40; color: #bbffde; }
.btn:active { transform: translateY(1px); }

table { width: 100%; border-collapse: collapse; }
th, td { text-align: left; padding: 8px 10px; border-bottom: 1px solid #202b38; vertical-align: top; }
th { color: #a9c3e1; font-weight: 700; position: sticky; top: 60px; background: #111825; }
tr:hover { background: #0f1520; }
.tag { background: #1c2736; display: inline-block; padding: 3px 8px; border-radius: 10px; margin: 2px; font-size: 12px; color: #c7d8ec; border: 1px solid #2b3a4f; }
.badge { padding: 2px 6px; border-radius: 8px; font-weight: 700; font-size: 12px; }
.badge.good { background: #103a28; color: #8ff0c1; border: 1px solid #1e5d42; }
.badge.warn { background: #3a2f10; color: #f8e38a; border: 1px solid #6e5a1a; }
.badge.bad  { background: #3a1010; color: #ffb4b4; border: 1px solid #5e1b1b; }
.mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace; font-size: 12px; color: #bcd0e6; }
textarea, input, select {
  width: 100%; background: #0e1420; color: var(--text); border: 1px solid #243140; border-radius: 8px;
  padding: 10px 12px; outline: none;
}
textarea:focus, input:focus { border-color: #3a5575; }
.job-catalog { display: grid; grid-template-columns: minmax(200px, 280px) 1fr; gap: 12px; }
@media (max-width: 900px) { .job-catalog { grid-template-columns: 1fr; } }
.job-list { max-height: 260px; overflow: auto; border: 1px solid #1f2b3b; border-radius: 10px; background: #0f1622; padding: 8px; }
.job-list button { width: 100%; text-align: left; margin-bottom: 6px; }
.job-preview { border: 1px solid #1f2b3b; border-radius: 10px; background: #0f1622; padding: 12px; min-height: 240px; }
.job-preview h3 { margin-top: 0; font-size: 14px; color: #cfe7ff; }
.job-stage { background: rgba(29,45,67,0.65); border: 1px solid #243349; border-radius: 8px; padding: 8px 10px; margin-bottom: 6px; }
.job-stage .title { font-weight: 600; }
.job-stage .meta { font-size: 12px; color: var(--muted2); margin-top: 4px; }
.inline-select { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
.inline-select > * { flex: 1 1 160px; }
.label-chip { font-size: 12px; color: var(--muted2); background: #162031; padding: 4px 8px; border-radius: 8px; border: 1px solid #253248; }
.sparkle { box-shadow: 0 0 18px rgba(110,193,255,0.25); filter: drop-shadow(0 0 6px rgba(110,193,255,0.45)); }
.topology-refresh { animation: fadeSlide 0.9s ease-in-out; }
@keyframes fadeSlide { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: translateY(0); } }
.topology-canvas svg { transition: transform 0.6s ease, opacity 0.6s ease; }
.node-core { stroke: #1a2533; stroke-width: 1.5px; transition: fill 0.6s ease, r 0.6s ease; }
.node-ring { fill: none; stroke-width: 3px; opacity: 0.85; transition: stroke 0.6s ease, stroke-dasharray 0.6s ease; }
.node-shadow { fill: none; stroke: #f39c12; stroke-width: 2px; opacity: 0.75; stroke-dasharray: 5 4; animation: halo 2s infinite ease-in-out; }
@keyframes halo { 0% { opacity: 0.1; } 50% { opacity: 0.8; } 100% { opacity: 0.1; } }
.link { stroke: #243140; stroke-width: 1.8px; stroke-linecap: round; opacity: 0.9; transition: stroke 0.6s ease, stroke-width 0.6s ease; }
.link.degraded { stroke: #f39c12; stroke-width: 2.4px; }
.link.down { stroke: #e74c3c; stroke-width: 2.6px; opacity: 0.95; }
.legend-dot.spark { box-shadow: 0 0 12px rgba(79,180,255,0.4); }
footer { color: var(--muted2); text-align: center; padding: 12px; }
hr { border: none; border-top: 1px solid #1f2a39; margin: 12px 0; }
.small { font-size: 12px; color: var(--muted2); }
.right { text-align: right; }
.flex { display: flex; gap: 8px; align-items: center; }
.remote-label { color: var(--muted); margin-top: 4px; }
.dag-wrap { margin-top: 12px; padding-top: 8px; border-top: 1px solid #1f2a39; }
.dag-row { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
.stage-card { background: var(--chip); border: 1px solid #2a3648; border-radius: 10px; padding: 10px 12px; min-width: 120px; }
.stage-card.bad { background: #2a1414; border-color: #5e1b1b; }
.stage-card .node { font-weight: 700; font-size: 14px; }
.stage-card .fmt { color: #8fbef6; font-size: 12px; margin-top: 4px; display: block; }
.stage-card .metrics { font-size: 12px; color: var(--muted2); margin-top: 4px; }
.stage-arrow { font-size: 18px; color: var(--muted2); }
.topology-canvas { width: 100%; height: 420px; position: relative; }
.topology-canvas svg { width: 100%; height: 100%; }
.topology-legend { margin-top: 8px; font-size: 12px; color: var(--muted2); display: flex; gap: 12px; flex-wrap: wrap; }
.legend-dot { width: 12px; height: 12px; border-radius: 50%; display: inline-block; }
.legend-dot.up { background: #2ecc71; border: 1px solid #1e5d42; }
.legend-dot.derate { background: #f1c40f; border: 1px solid #6e5a1a; }
.legend-dot.down { background: #e74c3c; border: 1px solid #5e1b1b; }
.legend-dot.assignment { border: 2px solid var(--accent); border-radius: 50%; width: 12px; height: 12px; box-shadow: 0 0 8px rgba(79,180,255,0.45); }
.legend-dot.fallback { border: 2px dashed #f39c12; background: transparent; }
.legend-dot.lossy { background: #f39c12; border: 1px solid #925208; }
.node-label { fill: #cfe7ff; font-size: 11px; pointer-events: none; text-anchor: middle; }
.link { stroke: #243140; stroke-width: 1.8px; stroke-linecap: round; opacity: 0.8; }
.node-core { stroke: #1a2533; stroke-width: 1.5px; }
.node-ring { fill: none; stroke-width: 3px; opacity: 0.8; }
.node-shadow { fill: none; stroke: #f39c12; stroke-width: 2px; opacity: 0.7; stroke-dasharray: 5 4; }
</style>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/d3@7"></script>
</head>
<body>
<header>
  <h1>Fabric Digital Twin — <small>Cluster Dashboard</small></h1>
  <div class="remote-label">Mode: __REMOTE_LABEL__</div>
</header>

<div class="container">

  <div class="card">
    <h2>Overview</h2>
    <div class="row">
      <div class="kpi">Nodes: <span id="k_nodes" class="v">—</span></div>
      <div class="kpi">Links: <span id="k_links" class="v">—</span></div>
      <div class="kpi">Federations: <span id="k_feds" class="v">—</span></div>
      <div class="kpi">Last Snapshot: <span id="k_ts" class="v">—</span></div>
      <div class="kpi">Max Fed Load: <span id="k_fedload" class="v">—</span></div>
      <div class="kpi">Plan Spread: <span id="k_spread" class="v">—</span></div>
      <div class="kpi">Fallback Coverage: <span id="k_resilience" class="v">—</span></div>
      <div class="kpi">Down: <span id="k_down" class="v">—</span></div>
      <div class="kpi">Reservations: <span id="k_resv" class="v">—</span></div>
      <div class="kpi">CPU used: <span id="k_cpu" class="v">—</span></div>
      <button class="btn" onclick="refresh()">Refresh</button>
      <button class="btn good" onclick="runDemo(true)">Dry-run Demo</button>
      <button class="btn primary" onclick="runDemo(false)">Reserve Demo</button>
    </div>
  </div>

  <div class="card">
    <h2>Fabric Topology</h2>
    <div class="topology-canvas" id="topology_canvas"><div class="small">Loading topology…</div></div>
    <div class="topology-legend">
      <span><span class="legend-dot up"></span> healthy</span>
      <span><span class="legend-dot derate"></span> thermal derate</span>
      <span><span class="legend-dot down"></span> down</span>
      <span><span class="legend-dot fallback"></span> fallback-ready</span>
      <span><span class="legend-dot assignment"></span> latest plan assignment</span>
      <span><span class="legend-dot lossy"></span> lossy / degraded link</span>
    </div>
  </div>

  <div class="grid-2">

    <div class="card">
      <h2>Nodes</h2>
      <div class="row" style="margin-bottom:8px;">
        <input id="nodeFilter" placeholder="Filter by name/arch/label..." oninput="renderNodes()" />
        <select id="archFilter" onchange="renderNodes()">
          <option value="">Arch: All</option>
          <option>x86_64</option>
          <option>arm64</option>
          <option>riscv64</option>
        </select>
      </div>
      <div style="max-height: 420px; overflow:auto;">
        <table>
          <thead>
            <tr>
              <th>Name / Class</th>
              <th>Arch & Formats</th>
              <th>CPU (free/max)</th>
              <th>Mem (free/max GB)</th>
              <th>VRAM (free/max GB)</th>
              <th>Health</th>
            </tr>
          </thead>
          <tbody id="nodes_tbody"></tbody>
        </table>
      </div>
      <div class="small">Tip: Health badges reflect <span class="mono">dyn.down</span>, <span class="mono">thermal_derate</span>, and recent reservations.</div>
    </div>

    <div class="card">
      <h2>Links</h2>
      <div style="max-height: 420px; overflow:auto;">
        <table>
          <thead>
            <tr>
              <th>Key</th>
              <th>Peers</th>
              <th>Speed (Gbps)</th>
              <th>RTT / Jitter (ms)</th>
              <th>Loss (%)</th>
              <th>ECN</th>
            </tr>
          </thead>
          <tbody id="links_tbody"></tbody>
        </table>
      </div>
    </div>

  </div>

  <div class="grid-2">

    <div class="card">
      <h2>Federations</h2>
      <div style="max-height: 320px; overflow:auto;">
        <table>
          <thead>
            <tr>
              <th>Name</th>
              <th>Load</th>
              <th>Down / Total</th>
              <th>Reservations</th>
              <th>Avg loss (%)</th>
              <th>Avg trust</th>
            </tr>
          </thead>
          <tbody id="feds_tbody"></tbody>
        </table>
      </div>
      <div class="small">Load blends CPU, memory, and accelerator utilisation across each federation.</div>
    </div>

    <div class="card">
      <h2>Federation Links</h2>
      <div style="max-height: 320px; overflow:auto;">
        <table>
          <thead>
            <tr>
              <th>A</th>
              <th>B</th>
              <th>Links</th>
              <th>Down</th>
              <th>Min speed (Gbps)</th>
              <th>Avg RTT (ms)</th>
              <th>Max loss (%)</th>
            </tr>
          </thead>
          <tbody id="fedlinks_tbody"></tbody>
        </table>
      </div>
      <div class="small">Aggregated cross-federation health synthesised from topology links and overrides.</div>
    </div>

  </div>

  <div class="grid-2">

    <div class="card">
      <h2>Plan a Job</h2>
      <div class="inline-select">
        <label class="flex"><input id="dryRun" type="checkbox" checked /> Dry run</label>
        <select id="strategySelect">
          <option value="greedy">Strategy: Greedy latency</option>
          <option value="cheapest-energy">Strategy: Cheapest energy</option>
          <option value="bandit">Strategy: Bandit format-aware</option>
          <option value="resilient">Strategy: Resilient federation</option>
          <option value="network-aware">Strategy: Network aware</option>
          <option value="federated">Strategy: Federated spread</option>
          <option value="balanced">Strategy: Balanced</option>
          <option value="rl-markov">Strategy: RL Markov planner</option>
        </select>
        <button class="btn primary" onclick="submitPlan()">Plan</button>
      </div>
      <div class="job-catalog" id="jobCatalogWrap">
        <div class="job-list" id="jobList">
          <div class="small">Loading job library…</div>
        </div>
        <div class="job-preview" id="jobPreview">
          <h3>Job preview</h3>
          <div class="small">Select a job from the catalog to inspect resources and load it into the planner form.</div>
        </div>
      </div>
      <div class="row" style="margin-top:10px; flex-wrap: wrap; gap:8px;">
        <button class="btn" onclick="planCatalog()">Plan entire catalog</button>
        <button class="btn secondary" onclick="loadSelectedJob()">Load selection into editor</button>
        <button class="btn" onclick="clearJobEditor()">Clear editor</button>
      </div>
      <textarea id="jobJson" rows="12" placeholder='Paste or edit the job JSON here…'></textarea>
      <div class="small">Jobs from the catalog include the <span class="mono">jobs/</span> YAML files. Edit the JSON to tweak resource requests, then run with any policy. Choose between greedy, bandit, and RL planners to compare live behaviour; the RL Markov planner minimises a Pareto-weighted cost and learns over time using the reinforcement learner.</div>
    </div>

    <div class="card">
      <h2>Recent Plans</h2>
      <div style="max-height: 420px; overflow:auto;">
        <table>
          <thead>
            <tr>
              <th>Job</th>
          <th>Latency (ms)</th>
          <th>Energy (kJ)</th>
          <th>Risk</th>
          <th>Reliability</th>
          <th>Spread / Federations</th>
          <th>Fallback Coverage</th>
          <th>Cross-fed Fallback</th>
          <th>Infeasible</th>
          <th>Stages</th>
            </tr>
          </thead>
          <tbody id="plans_tbody"></tbody>
        </table>
      </div>
      <div class="dag-wrap" id="plan_graph"></div>
    </div>

  </div>

  <div class="card">
    <h2>Chaos Runner</h2>
    <div class="row" style="margin-bottom:8px; flex-wrap:wrap; gap:8px;">
      <select id="chaosScenario" class="flex" style="min-width:200px;">
        <option value="">Loading scenarios…</option>
      </select>
      <label class="flex" style="gap:6px; align-items:center;"><span>Speed</span> <input id="chaosSpeed" type="number" min="0.1" step="0.1" value="12" style="width:90px;" /></label>
      <button class="btn" id="chaosStart" onclick="startChaos()">Start chaos</button>
      <button class="btn secondary" id="chaosStop" onclick="stopChaos()">Stop</button>
    </div>
    <div class="small" id="chaosStatus">Chaos idle. Loading scenarios…</div>
  </div>

  <div class="card">
    <h2>Apply Observation (node/link)</h2>
    <div class="row" style="margin-bottom:8px;">
      <button class="btn" onclick="applyObs()">Apply</button>
    </div>
    <textarea id="obsJson" rows="8" placeholder='{
  "payload": { "type": "node", "node": "ws-001", "changes": {"down": true, "thermal_derate": 0.3} }
}'></textarea>
    <div class="small">This mirrors the shape used by <span class="mono">sim/chaos.py</span> and merges into runtime <span class="mono">dyn</span> fields.</div>
  </div>

  <footer>Fabric DT — live simulator UI</footer>
</div>

<script>
let SNAP = null;
let LAST_PLAN = null;
let TOPO_SIM = null;
let TOPO_RESIZE = null;
let JOB_CATALOG = [];
let JOB_SELECTED = null;
let TOPO_SIGNATURE = null;
let TOPO_POS = new Map();
let CHAOS_OPTIONS = { scenarios: [], has_base: false, remote: false };
let CHAOS_STATUS_STATE = { running: false, remote: false };
let CHAOS_STATUS_TIMER = null;

async function fetchJSON(url, opts) {
  const r = await fetch(url, opts || {});
  const j = await r.json();
  if (!j.ok) throw new Error(j.error || 'request failed');
  return j.data;
}

function fmtPct(x) {
  if (x === null || x === undefined || Number.isNaN(Number(x))) return '—';
  return (Number(x) * 100).toFixed(0) + '%';
}
function fmt(x, d=2) {
  if (x === null || x === undefined || Number.isNaN(Number(x))) return '—';
  return Number(x).toFixed(d);
}
function ts(ms) { const d = new Date(ms); return d.toLocaleString(); }
function esc(text) {
  return String(text === undefined || text === null ? '' : text)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

function badge(s, cls) { return `<span class="badge ${cls}">${s}</span>`; }
function tag(s) { return `<span class="tag">${s}</span>`; }

async function loadJobCatalog() {
  try {
    const data = await fetchJSON('/api/jobs');
    JOB_CATALOG = Array.isArray(data) ? data : [];
  } catch (e) {
    console.warn('failed to load jobs', e);
    JOB_CATALOG = [];
  }
  if (JOB_CATALOG.length && (JOB_SELECTED === null || JOB_SELECTED >= JOB_CATALOG.length)) {
    JOB_SELECTED = 0;
  }
  renderJobCatalog();
}

function renderJobCatalog() {
  const listEl = document.getElementById('jobList');
  const previewEl = document.getElementById('jobPreview');
  if (!listEl || !previewEl) return;
  if (!JOB_CATALOG.length) {
    listEl.innerHTML = '<div class="small">No job YAMLs discovered yet. Drop files into <span class="mono">jobs/</span>.</div>';
    previewEl.innerHTML = '<h3>Job preview</h3><div class="small">No catalog entries available.</div>';
    JOB_SELECTED = null;
    return;
  }
  if (JOB_SELECTED === null) JOB_SELECTED = 0;
  listEl.innerHTML = JOB_CATALOG.map((entry, idx) => {
    const active = idx === JOB_SELECTED ? 'sparkle' : '';
    return `<button class="btn secondary ${active}" onclick="selectJob(${idx})">${esc(entry.id || 'job')} • ${esc(entry.file || '')}</button>`;
  }).join('');
  renderJobPreview(JOB_CATALOG[JOB_SELECTED]);
  loadSelectedJob(true);
}

function renderJobPreview(entry) {
  const previewEl = document.getElementById('jobPreview');
  if (!previewEl) return;
  if (!entry) {
    previewEl.innerHTML = '<h3>Job preview</h3><div class="small">Select a job to inspect stages and resources.</div>';
    return;
  }
  const job = entry.job || {};
  const stages = Array.isArray(job.stages) ? job.stages : [];
  let html = `<h3>${esc(entry.id || 'job')}</h3>`;
  html += `<div class="label-chip">File: ${esc(entry.file || '')}</div>`;
  if (job.deadline_ms) {
    html += `<div class="label-chip">Deadline: ${fmt(job.deadline_ms, 0)} ms</div>`;
  }
  if (!stages.length) {
    html += '<div class="small">This job has no stages defined.</div>';
    previewEl.innerHTML = html;
    return;
  }
  html += stages.map((stage, idx) => {
    const res = stage.resources || {};
    const formats = Array.isArray(stage.allowed_formats) ? stage.allowed_formats.join(', ') : 'any';
    const cpu = res.cpu_cores !== undefined ? res.cpu_cores : '—';
    const mem = res.mem_gb !== undefined ? res.mem_gb : '—';
    const vram = res.gpu_vram_gb !== undefined ? res.gpu_vram_gb : '—';
    const size = stage.size_mb !== undefined ? `${stage.size_mb} MB` : '—';
    return `
      <div class="job-stage">
        <div class="title">Stage ${idx + 1}: ${esc(stage.id || stage.type || 'stage')}</div>
        <div class="meta">Type: ${esc(stage.type || '—')} • Size: ${esc(size)}</div>
        <div class="meta">CPU: ${esc(cpu)} cores • Mem: ${esc(mem)} GB • VRAM: ${esc(vram)} GB</div>
        <div class="meta">Formats: ${esc(formats)}</div>
      </div>
    `;
  }).join('');
  previewEl.innerHTML = html;
}

function selectJob(idx) {
  if (idx < 0 || idx >= JOB_CATALOG.length) return;
  JOB_SELECTED = idx;
  renderJobCatalog();
}

function loadSelectedJob(auto=false) {
  if (JOB_SELECTED === null || !JOB_CATALOG[JOB_SELECTED]) {
    if (!auto) alert('Select a job from the catalog first.');
    return;
  }
  const job = JOB_CATALOG[JOB_SELECTED].job || {};
  const editor = document.getElementById('jobJson');
  if (editor) {
    editor.value = JSON.stringify(job, null, 2);
    if (!auto) editor.focus();
  }
}

function clearJobEditor() {
  const editor = document.getElementById('jobJson');
  if (editor) editor.value = '';
}

function renderChaosControls() {
  const select = document.getElementById('chaosScenario');
  const statusEl = document.getElementById('chaosStatus');
  const startBtn = document.getElementById('chaosStart');
  const stopBtn = document.getElementById('chaosStop');
  const speedInput = document.getElementById('chaosSpeed');
  if (!select || !statusEl) return;
  const remote = Boolean((CHAOS_OPTIONS && CHAOS_OPTIONS.remote) || (CHAOS_STATUS_STATE && CHAOS_STATUS_STATE.remote));
  const scenarios = Array.isArray(CHAOS_OPTIONS.scenarios) ? CHAOS_OPTIONS.scenarios : [];
  const items = [];
  if (CHAOS_OPTIONS.has_base) {
    items.push({ value: '', label: 'Default schedule' });
  }
  scenarios.forEach(sc => {
    if (!sc || !sc.name) return;
    const desc = sc.description ? ` – ${sc.description}` : '';
    items.push({ value: sc.name, label: `Scenario: ${sc.name}${desc}` });
  });
  if (!items.length) {
    select.innerHTML = '<option value="" disabled>No scenarios available</option>';
    select.value = '';
    select.disabled = true;
  } else {

    select.innerHTML = items.map(opt => `<option value="${esc(opt.value)}">${esc(opt.label)}</option>`).join('');
    const current = CHAOS_STATUS_STATE && CHAOS_STATUS_STATE.scenario !== undefined ? CHAOS_STATUS_STATE.scenario : null;
    if (current !== null && select.querySelector(`option[value="${current}"]`)) {
      select.value = current;
    } else if (CHAOS_STATUS_STATE && CHAOS_STATUS_STATE.scenario === null) {
      select.value = '';
    }
    select.disabled = false;
  }
  if (remote) {
    if (startBtn) startBtn.disabled = true;
    if (stopBtn) stopBtn.disabled = true;
    if (speedInput) speedInput.disabled = true;
    statusEl.textContent = 'Chaos controls require the embedded DT runtime.';
    return;
  }
  if (CHAOS_OPTIONS && CHAOS_OPTIONS.error) {
    statusEl.textContent = `Chaos scenarios unavailable: ${CHAOS_OPTIONS.error}`;
    if (startBtn) startBtn.disabled = true;
    if (stopBtn) stopBtn.disabled = true;
    if (speedInput) speedInput.disabled = true;
    return;
  }
  if (startBtn) startBtn.disabled = Boolean(CHAOS_STATUS_STATE && CHAOS_STATUS_STATE.running);
  if (stopBtn) stopBtn.disabled = !(CHAOS_STATUS_STATE && CHAOS_STATUS_STATE.running);
  if (speedInput) speedInput.disabled = false;
  if (CHAOS_STATUS_STATE && CHAOS_STATUS_STATE.last_error) {
    statusEl.textContent = `Last error: ${CHAOS_STATUS_STATE.last_error}`;
    return;
  }
  if (CHAOS_STATUS_STATE && CHAOS_STATUS_STATE.running) {
    const scenarioLabel = CHAOS_STATUS_STATE.scenario ? `Scenario: ${CHAOS_STATUS_STATE.scenario}` : 'Default schedule';
    const speed = CHAOS_STATUS_STATE.speed !== undefined ? fmt(CHAOS_STATUS_STATE.speed, 1) : '—';
    const events = CHAOS_STATUS_STATE.events !== undefined ? CHAOS_STATUS_STATE.events : '—';
    statusEl.textContent = `Chaos running • ${scenarioLabel} • speed x${speed} • events ${events}`;
    return;
  }
  if (items.length) {
    statusEl.textContent = 'Chaos idle. Pick a scenario and start to inject failures.';
  } else {
    statusEl.textContent = 'Chaos idle. Define chaos schedules in sim/topology.yaml to enable experiments.';
  }
}

async function loadChaosOptions() {
  try {
    const data = await fetchJSON('/api/chaos_scenarios');
    CHAOS_OPTIONS = Object.assign({ scenarios: [], has_base: false, remote: false }, data || {});
    if (!Array.isArray(CHAOS_OPTIONS.scenarios)) CHAOS_OPTIONS.scenarios = [];
  } catch (e) {
    CHAOS_OPTIONS = { scenarios: [], has_base: false, error: e.message, remote: false };
  }
  renderChaosControls();
}

async function refreshChaosStatus() {
  try {
    const data = await fetchJSON('/api/chaos');
    CHAOS_STATUS_STATE = Object.assign({ running: false }, data || {});
  } catch (e) {
    CHAOS_STATUS_STATE = { running: false, error: e.message };
  }
  renderChaosControls();
}

async function startChaos() {
  const select = document.getElementById('chaosScenario');
  const speedInput = document.getElementById('chaosSpeed');
  const scenario = select ? select.value : '';
  const speed = speedInput ? Number(speedInput.value || 12) : 12;
  try {
    await fetchJSON('/api/chaos', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ action: 'start', scenario: scenario || null, speed })
    });
    await refreshChaosStatus();
    await refresh();
  } catch (e) {
    alert('Failed to start chaos: ' + e.message);
  }
}

async function stopChaos() {
  try {
    await fetchJSON('/api/chaos', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ action: 'stop' })
    });
    await refreshChaosStatus();
  } catch (e) {
    alert('Failed to stop chaos: ' + e.message);
  }
}

function renderOverview() {
  if (!SNAP) return;
  const nodes = SNAP.nodes || [];
  const links = SNAP.links || [];
  const federations = SNAP.federations || [];
  document.getElementById('k_nodes').textContent = nodes.length;
  document.getElementById('k_links').textContent = links.length;
  document.getElementById('k_ts').textContent = ts(SNAP.ts);
  const kFeds = document.getElementById('k_feds');
  if (kFeds) kFeds.textContent = federations.length;
  const down = nodes.filter(n => (n.dyn||{}).down).length;
  const reservations = nodes.reduce((acc, n) => {
    const dyn = n.dyn || {};
    const res = dyn.reservations ? Object.keys(dyn.reservations).length : 0;
    return acc + res;
  }, 0);
  let usedCpu = 0;
  let maxCpu = 0;
  nodes.forEach(n => {
    const eff = n.effective || {};
    const maxC = Number(eff.max_cpu_cores || 0);
    const free = Number(eff.free_cpu_cores || 0);
    if (maxC > 0) {
      usedCpu += Math.max(0, maxC - free);
      maxCpu += maxC;
    }
  });
  const maxFedLoad = federations.length ? Math.max(...federations.map(fed => Number(fed.load_factor || 0))) : null;
  const fedLoadEl = document.getElementById('k_fedload');
  if (fedLoadEl) fedLoadEl.textContent = maxFedLoad !== null ? fmtPct(maxFedLoad) : '—';
  const spread = LAST_PLAN && LAST_PLAN.federation_spread !== undefined && LAST_PLAN.federation_spread !== null ? LAST_PLAN.federation_spread : null;
  const spreadEl = document.getElementById('k_spread');
  if (spreadEl) spreadEl.textContent = spread !== null ? fmt(spread, 2) : '—';
  const resilience = LAST_PLAN && LAST_PLAN.resilience_score !== undefined && LAST_PLAN.resilience_score !== null ? LAST_PLAN.resilience_score : null;
  const resilienceEl = document.getElementById('k_resilience');
  if (resilienceEl) resilienceEl.textContent = resilience !== null ? fmtPct(resilience) : '—';
  document.getElementById('k_down').textContent = down;
  document.getElementById('k_resv').textContent = reservations;
  document.getElementById('k_cpu').textContent = maxCpu > 0 ? fmtPct(usedCpu / maxCpu) : '—';
}

function renderNodes() {
  if (!SNAP) return;
  const q = (document.getElementById('nodeFilter').value || '').toLowerCase();
  const arch = document.getElementById('archFilter').value || '';
  const tb = document.getElementById('nodes_tbody');
  tb.innerHTML = '';
  const fedMap = SNAP.node_federations || {};
  SNAP.nodes.forEach(n => {
    const eff = n.effective || {};
    const dyn = n.dyn || {};
    const labels = n.labels || {};
    const archMatch = (!arch || (n.arch||'')===arch);
    const fed = fedMap[n.name] || labels.federation || labels.zone || '';
    const hay = (n.name+' '+(n.class||'')+' '+(n.arch||'')+' '+fed+' '+JSON.stringify(labels)).toLowerCase();
    if (!archMatch) return;
    if (q && !hay.includes(q)) return;

    let health = '';
    if (dyn.down) health += badge('DOWN', 'bad') + ' ';
    if (dyn.thermal_derate && dyn.thermal_derate>0) health += badge('DERATE '+Math.round(dyn.thermal_derate*100)+'%', 'warn') + ' ';
    const resv = (dyn.reservations && Object.keys(dyn.reservations).length) ? Object.keys(dyn.reservations).length : 0;
    if (resv>0) health += badge(`${resv} resv`, 'good');

    const fmts = (n.formats_supported||[]).map(tag).join(' ');
    let labelPairs = Object.entries(labels);
    if (fed) {
      labelPairs = [["federation", fed], ...labelPairs];
    }
    const lbls = labelPairs.map(([k,v]) => tag(`${k}:${v}`)).join(' ');

    tb.insertAdjacentHTML('beforeend', `
      <tr>
        <td><div class="mono">${n.name}</div><div class="small">${n.class||''}</div></td>
        <td><div>${n.arch||'—'}</div><div>${fmts||''}</div></td>
        <td><div>${fmt(eff.free_cpu_cores,2)} / ${fmt(eff.max_cpu_cores,2)}</div></td>
        <td><div>${fmt(eff.free_mem_gb,2)} / ${fmt(eff.max_mem_gb,2)}</div></td>
        <td><div>${fmt(eff.free_gpu_vram_gb,2)} / ${fmt(eff.max_gpu_vram_gb,2)}</div></td>
        <td>${health||''}<div class="small">${lbls||''}</div></td>
      </tr>
    `);
  });
}

function renderLinks() {
  if (!SNAP) return;
  const tb = document.getElementById('links_tbody');
  tb.innerHTML = '';
  SNAP.links.forEach(l => {
    const e = l.effective || {};
    tb.insertAdjacentHTML('beforeend', `
      <tr>
        <td class="mono">${l.key}</td>
        <td>${l.a} ↔ ${l.b}</td>
        <td>${fmt(e.speed_gbps,2)}</td>
        <td>${fmt(e.rtt_ms,1)} / ${fmt(e.jitter_ms,1)}</td>
        <td>${fmt(e.loss_pct,2)}</td>
        <td>${e.ecn ? 'Yes' : 'No'}</td>
      </tr>
    `);
  });
}

function renderFederations() {
  const tb = document.getElementById('feds_tbody');
  if (!tb) return;
  tb.innerHTML = '';
  const feds = SNAP && SNAP.federations ? SNAP.federations : [];
  if (!feds.length) {
    tb.innerHTML = '<tr><td colspan="6" class="small">No federation data.</td></tr>';
    return;
  }
  feds.forEach(fed => {
    const nodes = (fed.nodes || []).join(', ');
    tb.insertAdjacentHTML('beforeend', `
      <tr>
        <td><div class="mono">${fed.name || '—'}</div><div class="small">${nodes || '—'}</div></td>
        <td>${fmtPct(fed.load_factor)}</td>
        <td>${fed.down_nodes || 0} / ${(fed.nodes || []).length}</td>
        <td>${fed.reservations || 0}</td>
        <td>${fed.avg_loss_pct !== null && fed.avg_loss_pct !== undefined ? fmt(fed.avg_loss_pct, 2) : '—'}</td>
        <td>${fed.avg_trust !== null && fed.avg_trust !== undefined ? fmt(fed.avg_trust, 2) : '—'}</td>
      </tr>
    `);
  });
}

function renderFederationLinks() {
  const tb = document.getElementById('fedlinks_tbody');
  if (!tb) return;
  tb.innerHTML = '';
  const edges = SNAP && SNAP.federation_links ? SNAP.federation_links : [];
  if (!edges.length) {
    tb.innerHTML = '<tr><td colspan="7" class="small">No federation links available.</td></tr>';
    return;
  }
  edges.forEach(e => {
    tb.insertAdjacentHTML('beforeend', `
      <tr>
        <td>${e.a || '—'}</td>
        <td>${e.b || '—'}</td>
        <td>${e.links || 0}</td>
        <td>${e.down_links || 0}</td>
        <td>${e.min_speed_gbps !== null && e.min_speed_gbps !== undefined ? fmt(Number(e.min_speed_gbps), 2) : '—'}</td>
        <td>${e.avg_rtt_ms !== null && e.avg_rtt_ms !== undefined ? fmt(Number(e.avg_rtt_ms), 2) : '—'}</td>
        <td>${e.max_loss_pct !== null && e.max_loss_pct !== undefined ? fmt(Number(e.max_loss_pct), 2) : '—'}</td>
      </tr>
    `);
  });
}

function destroyTopology() {
  if (TOPO_SIM) {
    TOPO_SIM.stop();
    TOPO_SIM = null;
  }
}

function renderTopology() {
  const wrap = document.getElementById('topology_canvas');
  if (!wrap) return;
  if (!SNAP || !Array.isArray(SNAP.nodes) || SNAP.nodes.length === 0) {
    destroyTopology();
    wrap.innerHTML = '<div class="small">No topology data yet.</div>';
    TOPO_SIGNATURE = null;
    TOPO_POS = new Map();
    return;
  }

  const assignments = new Map();
  if (LAST_PLAN && Array.isArray(LAST_PLAN.per_stage)) {
    LAST_PLAN.per_stage.forEach(stage => {
      if (!stage || stage.infeasible || !stage.node) return;
      const prev = assignments.get(stage.node) || {count: 0, stages: []};
      prev.count += 1;
      if (stage.id) prev.stages.push(stage.id);
      assignments.set(stage.node, prev);
    });
  }

  const fallbackAssignments = new Map();
  if (LAST_PLAN && Array.isArray(LAST_PLAN.per_stage)) {
    LAST_PLAN.per_stage.forEach(stage => {
      if (!stage || !Array.isArray(stage.fallbacks)) return;
      stage.fallbacks.forEach(entry => {
        const name = typeof entry === 'string' ? entry : (entry && entry.node) || null;
        if (!name) return;
        const prev = fallbackAssignments.get(name) || {count: 0, stages: []};
        prev.count += 1;
        if (stage.id) prev.stages.push(stage.id);
        fallbackAssignments.set(name, prev);
      });
    });
  }

  const fedMap = SNAP.node_federations || {};
  const nodes = SNAP.nodes.map(n => {
    const eff = n.effective || {};
    const dyn = n.dyn || {};
    const maxCpu = Number(eff.max_cpu_cores || 0);
    const freeCpu = Number(eff.free_cpu_cores || 0);
    const maxMem = Number(eff.max_mem_gb || 0);
    const freeMem = Number(eff.free_mem_gb || 0);
    const res = dyn.reservations ? Object.keys(dyn.reservations).length : 0;
    const assign = assignments.get(n.name);
    const fallback = fallbackAssignments.get(n.name);
    return {
      id: n.name,
      arch: n.arch,
      class: n.class,
      labels: n.labels || {},
      down: Boolean(dyn.down),
      derate: Number(dyn.thermal_derate || 0),
      cpuCap: maxCpu,
      cpuUsed: Math.max(0, maxCpu - freeCpu),
      memCap: maxMem,
      memUsed: Math.max(0, maxMem - freeMem),
      reservations: res,
      assignCount: assign ? assign.count : 0,
      assignStages: assign ? assign.stages : [],
      shadowCount: fallback ? fallback.count : 0,
      shadowStages: fallback ? fallback.stages : [],
      federation: fedMap[n.name] || (n.labels || {}).federation || (n.labels || {}).zone || '—',
    };
  });

  const lookup = new Map(nodes.map(n => [n.id, n]));
  const links = (SNAP.links || [])
    .filter(l => lookup.has(l.a) && lookup.has(l.b))
    .map(l => {
      const eff = l.effective || {};
      return {
        source: l.a,
        target: l.b,
        down: Boolean(eff.down),
        speed: Number(eff.speed_gbps || 0),
        loss: Number(eff.loss_pct || 0),
        rtt: Number(eff.rtt_ms || 0),
        jitter: Number(eff.jitter_ms || 0),
        key: l.key,
      };
    });

  const round = x => Math.round((Number(x) || 0) * 1000) / 1000;
  const nodeParts = nodes
    .map(n => [
      n.id,
      n.down ? 1 : 0,
      round(n.derate),
      round(n.cpuCap),
      round(n.cpuUsed),
      round(n.memCap),
      round(n.memUsed),
      n.reservations,
      n.assignCount,
      n.assignStages.join(','),
      n.shadowCount,
      n.shadowStages.join(','),
      n.federation || ''
    ].join('|'))
    .sort();
  const linkParts = links
    .map(l => [
      l.source,
      l.target,
      l.down ? 1 : 0,
      round(l.speed),
      round(l.loss),
      round(l.rtt),
      round(l.jitter)
    ].join('|'))
    .sort();
  const signature = `${nodeParts.join(';')}#${linkParts.join(';')}`;
  if (signature === TOPO_SIGNATURE) {
    return;
  }
  TOPO_SIGNATURE = signature;

  const prevPositions = new Map();
  if (TOPO_SIM && typeof TOPO_SIM.nodes === 'function') {
    try {
      TOPO_SIM.nodes().forEach(n => {
        if (n && n.id) prevPositions.set(n.id, {x: n.x, y: n.y});
      });
    } catch (e) {
      // ignore if simulation not ready
    }
  } else if (TOPO_POS && typeof TOPO_POS.forEach === 'function') {
    TOPO_POS.forEach((value, key) => {
      if (value && typeof value.x === 'number' && typeof value.y === 'number') {
        prevPositions.set(key, {x: value.x, y: value.y});
      }
    });
  }

  destroyTopology();
  TOPO_POS = new Map(prevPositions);

  const width = wrap.clientWidth || 720;
  const height = Math.max(360, Math.min(760, 180 + nodes.length * 14));
  wrap.innerHTML = '';

  nodes.forEach(n => {
    const prev = prevPositions.get(n.id);
    if (prev) {
      n.x = prev.x;
      n.y = prev.y;
    }
  });

  const svg = d3
    .select(wrap)
    .append('svg')
    .attr('viewBox', `0 0 ${width} ${height}`)
    .attr('preserveAspectRatio', 'xMidYMid meet');

  const g = svg.append('g');

  const linkWidth = l => 1.5 + Math.log1p(Math.max(0.2, l.speed));
  const linkColor = l => {
    if (l.down) return '#e74c3c';
    if (l.loss >= 2.0 || l.jitter >= 2.0) return '#f39c12';
    return '#2c3f57';
  };

  const linkGroup = g.append('g').attr('class', 'links');
  const link = linkGroup
    .selectAll('line')
    .data(links)
    .enter()
    .append('line')
    .attr('class', d => {
      if (d.down) return 'link down';
      if (d.loss >= 2.0 || d.jitter >= 2.0) return 'link degraded';
      return 'link';
    })
    .attr('stroke', linkColor)
    .attr('stroke-width', linkWidth);

  link.append('title').text(l => {
    const parts = [
      `${l.source} ↔ ${l.target}`,
      `speed: ${fmt(l.speed, 2)} Gbps`,
      `rtt: ${fmt(l.rtt, 1)} ms`,
      `loss: ${fmt(l.loss, 2)} %`,
    ];
    if (l.down) parts.push('status: DOWN');
    return parts.join('\n');
  });

  const nodeRadius = d => {
    const cpu = Math.max(0, d.cpuCap);
    const mem = Math.max(0, d.memCap);
    const cpuTerm = Math.log10(cpu + 1) * 10;
    const memTerm = Math.log10(mem + 1) * 4;
    return 12 + Math.min(22, cpuTerm + memTerm);
  };

  const nodeColor = d => {
    if (d.down) return '#e74c3c';
    if (d.derate > 0.01) return '#f1c40f';
    const pct = d.cpuCap > 0 ? Math.min(1, d.cpuUsed / d.cpuCap) : 0;
    return d3.interpolateBlues(0.3 + pct * 0.6);
  };

  const simulation = d3
    .forceSimulation(nodes)
    .force(
      'link',
      d3
        .forceLink(links)
        .id(d => d.id)
        .distance(l => {
          const base = 160;
          const speed = Math.max(0.2, l.speed || 0.2);
          return base / Math.sqrt(speed);
        })
        .strength(0.6)
    )
    .force('charge', d3.forceManyBody().strength(-260))
    .force('center', d3.forceCenter(width / 2, height / 2))
    .force('collision', d3.forceCollide().radius(d => nodeRadius(d) + 16));

  const drag = sim => {
    function dragstarted(event, d) {
      if (!event.active) sim.alphaTarget(0.2).restart();
      d.fx = d.x;
      d.fy = d.y;
    }
    function dragged(event, d) {
      d.fx = event.x;
      d.fy = event.y;
    }
    function dragended(event, d) {
      if (!event.active) sim.alphaTarget(0);
      d.fx = null;
      d.fy = null;
    }
    return d3.drag().on('start', dragstarted).on('drag', dragged).on('end', dragended);
  };

  const nodesGroup = g.append('g').attr('class', 'nodes');
  const node = nodesGroup
    .selectAll('g')
    .data(nodes)
    .enter()
    .append('g')
    .attr('class', d => (d.assignCount > 0 ? 'node-group sparkle' : 'node-group'))
    .call(drag(simulation));

  node
    .append('circle')
    .attr('class', 'node-shadow')
    .attr('r', d => nodeRadius(d) + 9)
    .style('display', d => (d.shadowCount > 0 ? 'block' : 'none'));

  node
    .append('circle')
    .attr('class', d => (d.assignCount > 0 ? 'node-ring sparkle' : 'node-ring'))
    .attr('r', d => nodeRadius(d) + 5)
    .attr('stroke', d => (d.assignCount > 0 ? '#6fc1ff' : '#2ecc71'))
    .attr('stroke-dasharray', d => (d.assignCount > 0 ? null : '6 4'))
    .style('display', d => (d.assignCount > 0 || d.reservations > 0 ? 'block' : 'none'));

  node
    .append('circle')
    .attr('class', 'node-core')
    .attr('r', d => nodeRadius(d))
    .attr('fill', nodeColor);

  node
    .append('text')
    .attr('class', 'node-label')
    .attr('dy', 4)
    .text(d => d.id);

  node.append('title').text(d => {
    const status = d.down
      ? 'status: DOWN'
      : d.derate > 0.01
      ? `thermal derate: ${fmtPct(Math.min(1, d.derate))}`
      : 'status: healthy';
    const res = `reservations: ${d.reservations}`;
    const assign = d.assignCount
      ? `latest plan stages: ${d.assignStages.join(', ')}`
      : 'latest plan stages: none';
    const fallback = d.shadowCount
      ? `fallback stages: ${d.shadowStages.join(', ')}`
      : 'fallback stages: none';
    return [
      d.id,
      `${d.arch || '—'} ${d.class || ''}`.trim(),
      status,
      `cpu ${fmt(d.cpuUsed, 1)} / ${fmt(d.cpuCap, 1)} cores`,
      `mem ${fmt(d.memUsed, 1)} / ${fmt(d.memCap, 1)} GB`,
      `federation: ${d.federation}`,
      res,
      assign,
      fallback,
    ]
      .filter(Boolean)
      .join('\n');
  });

  simulation.on('tick', () => {
    link
      .attr('x1', d => d.source.x)
      .attr('y1', d => d.source.y)
      .attr('x2', d => d.target.x)
      .attr('y2', d => d.target.y);
    node.attr('transform', d => `translate(${d.x},${d.y})`);
    const latest = new Map();
    nodes.forEach(n => {
      if (n && n.id) {
        latest.set(n.id, {x: n.x, y: n.y});
      }
    });
    TOPO_POS = latest;
  });

  TOPO_SIM = simulation;
}


function renderPlanGraph() {
  const wrap = document.getElementById('plan_graph');
  if (!wrap) return;
  if (!LAST_PLAN) {
    wrap.innerHTML = '<div class="small">No plans yet.</div>';
    return;
  }
  const per = LAST_PLAN.per_stage || [];
  if (!per.length) {
    wrap.innerHTML = '<div class="small">Plan contains no stages.</div>';
    return;
  }
  const parts = per.map(s => {
    const node = s.node || '—';
    const fmtBadge = s.format ? `<span class="fmt">${s.format}</span>` : '';
    const cls = s.infeasible ? 'stage-card bad' : 'stage-card';
    const relMetric = s.reliability !== undefined && s.reliability !== null ? ` • rel:${fmt(s.reliability,2)}` : '';
    const availMetric = s.availability_window_sec !== undefined && s.availability_window_sec !== null ? ` • avail:${fmt(s.availability_window_sec,2)}s` : '';
    const metrics = `<div class="metrics">c:${fmt(s.compute_ms,1)} ms • x:${fmt(s.xfer_ms,1)} ms${relMetric}${availMetric}</div>`;
    const reason = s.infeasible && s.reason ? `<div class="small">${s.reason}</div>` : '';
    const badgeHtml = s.infeasible ? `<div class="badge bad">Blocked</div>` : '';
    const fallbackNodes = Array.isArray(s.fallbacks) ? s.fallbacks : [];
    const fallbackFeds = Array.isArray(s.fallback_federations) ? s.fallback_federations : [];
    const fallbackPairs = fallbackNodes.map((entry, idx) => {
      const name = typeof entry === 'string' ? entry : (entry && entry.node) || '—';
      const rel = entry && typeof entry === 'object' && entry.reliability !== undefined && entry.reliability !== null
        ? ` rel:${fmt(entry.reliability,2)}`
        : '';
      const avail = entry && typeof entry === 'object' && entry.availability_window_sec !== undefined && entry.availability_window_sec !== null
        ? ` avail:${fmt(entry.availability_window_sec,2)}s`
        : '';
      const fed = fallbackFeds[idx];
      const fedTag = fed ? tag(`fed:${fed}`) : '';
      return `<span class="mono">${name}</span>${fedTag}${rel}${avail}`;
    }).join(' ');
    const fallback = fallbackPairs
      ? `<div class="small">Fallback: ${fallbackPairs}</div>`
      : '';
    const fed = s.federation ? `<div class="small">Federation: ${s.federation}</div>` : '';
    return `<div class="${cls}"><div class="mono">${s.id||'?'}</div><div class="node">${node}</div>${fmtBadge}${metrics}${fallback}${fed}${badgeHtml}${reason}</div>`;
  }).join('<div class="stage-arrow">→</div>');
  wrap.innerHTML = `<div class="dag-row">${parts}</div>`;
  const spreadStr = LAST_PLAN && LAST_PLAN.federation_spread !== undefined && LAST_PLAN.federation_spread !== null ? ` • Spread ${fmt(LAST_PLAN.federation_spread,2)}` : '';
  const resilienceStr = LAST_PLAN && LAST_PLAN.resilience_score !== undefined && LAST_PLAN.resilience_score !== null ? ` • Resilience ${fmtPct(LAST_PLAN.resilience_score)}` : '';
  const reliabilityStr = LAST_PLAN && LAST_PLAN.avg_reliability !== undefined && LAST_PLAN.avg_reliability !== null ? ` • Reliability ${fmt(LAST_PLAN.avg_reliability,2)}` : '';
  wrap.insertAdjacentHTML('beforeend', `<div class="small" style="margin-top:6px;">Latency ${fmt(LAST_PLAN.latency_ms,1)} ms • Energy ${fmt(LAST_PLAN.energy_kj,3)} kJ • Risk ${fmt(LAST_PLAN.risk,3)}${spreadStr}${resilienceStr}${reliabilityStr}</div>`);
}

function renderPlans() {
  const tb = document.getElementById('plans_tbody');
  tb.innerHTML = '';
  fetchJSON('/api/plans').then(data => {
    LAST_PLAN = data.length ? data[0] : null;
    data.forEach(p => {
      const stages = (p.per_stage||[]).map(s => {
        const nodeName = s.node || '—';
        const fmtTag = s.format ? tag(s.format) : '';
        const inf = s.infeasible ? badge('X','bad') : '';
        const fallbackNodes = Array.isArray(s.fallbacks) ? s.fallbacks : [];
        const fallbackFeds = Array.isArray(s.fallback_federations) ? s.fallback_federations : [];
        const fallbackPairs = fallbackNodes.map((entry, idx) => {
          const name = typeof entry === 'string' ? entry : (entry && entry.node) || '—';
          const rel = entry && typeof entry === 'object' && entry.reliability !== undefined && entry.reliability !== null
            ? ` rel:${fmt(entry.reliability,2)}`
            : '';
          const avail = entry && typeof entry === 'object' && entry.availability_window_sec !== undefined && entry.availability_window_sec !== null
            ? ` avail:${fmt(entry.availability_window_sec,2)}s`
            : '';
          const fed = fallbackFeds[idx];
          const fedTag = fed ? tag(`fed:${fed}`) : '';
          return `<span class="mono">${name}</span>${fedTag}${rel}${avail}`;
        }).join(' ');
        const fallbackHtml = fallbackPairs ? `<div class="small">Fallback: ${fallbackPairs}</div>` : '';
        const fedTag = s.federation ? `<div class="small">Federation: ${s.federation}</div>` : '';
        const reason = s.infeasible && s.reason ? `<div class="small">${s.reason}</div>` : '';
        const load = s.load_factor !== undefined && s.load_factor !== null ? `<span class="mono">load:${fmt(s.load_factor,2)}</span>` : '';
        const net = s.network_penalty !== undefined && s.network_penalty !== null ? `<span class="mono">net:${fmt(s.network_penalty,2)}</span>` : '';
        const score = s.expected_cost !== undefined && s.expected_cost !== null ? `<span class="mono">J:${fmt(s.expected_cost,3)}</span>` : '';
        const relMetric = s.reliability !== undefined && s.reliability !== null ? `<span class="mono">rel:${fmt(s.reliability,2)}</span>` : '';
        const availMetric = s.availability_window_sec !== undefined && s.availability_window_sec !== null ? `<span class="mono">avail:${fmt(s.availability_window_sec,2)}s</span>` : '';
        const extras = [load, net, score, relMetric, availMetric].filter(Boolean).join(' ');
        const metrics = `<span class="mono">c:${fmt(s.compute_ms,1)}ms</span> <span class="mono">x:${fmt(s.xfer_ms,1)}ms</span>`;
        const extraLine = extras ? `<div class="small">${extras}</div>` : '';
        return `<div class="small"><span class="mono">${s.id||'?'}</span> → <b>${nodeName}</b> ${fmtTag} ${inf} ${metrics}${extraLine}${fallbackHtml}${fedTag}${reason}<\/div>`;
      }).join('');
      const spreadVal = p.federation_spread !== null && p.federation_spread !== undefined ? fmt(p.federation_spread, 2) : '—';
      const feds = (p.federations_in_use || []).map(tag).join(' ');
      const spreadCell = `${spreadVal}${feds ? `<div class="small">${feds}</div>` : ''}`;
      const resilienceVal = p.resilience_score !== null && p.resilience_score !== undefined ? fmtPct(p.resilience_score) : '—';
      const crossVal = p.cross_federation_fallback_ratio !== null && p.cross_federation_fallback_ratio !== undefined ? fmtPct(p.cross_federation_fallback_ratio) : '—';
      const reliabilityVal = p.avg_reliability !== null && p.avg_reliability !== undefined ? fmt(p.avg_reliability,2) : '—';
      tb.insertAdjacentHTML('beforeend', `
        <tr>
          <td><div class="mono">${p.job_id||'—'}</div><div class="small">${p.strategy||'greedy'} ${p.dry_run?'(dry)':''}</div></td>
          <td>${fmt(p.latency_ms,1)}</td>
          <td>${fmt(p.energy_kj,3)}</td>
          <td>${fmt(p.risk,3)}</td>
          <td>${reliabilityVal}</td>
          <td>${spreadCell}</td>
          <td>${resilienceVal}</td>
          <td>${crossVal}</td>
          <td>${p.infeasible ? badge('Yes','bad') : badge('No','good')}</td>
          <td>${stages}</td>
        </tr>
      `);
    });
    renderPlanGraph();
    renderTopology();
    renderOverview();
  }).catch(e => {
    tb.innerHTML = `<tr><td colspan="9" class="small">No plans yet.</td></tr>`;
    LAST_PLAN = null;
    renderPlanGraph();
    renderTopology();
    renderOverview();
  });
}

async function refresh() {
  try {
    const data = await fetchJSON('/api/snapshot');
    SNAP = data;
    renderOverview();
    renderNodes();
    renderLinks();
    renderFederations();
    renderFederationLinks();
    renderTopology();
    renderPlans();
  } catch (e) {
    console.error(e);
    alert('Failed to refresh snapshot: '+e.message);
  }
}

async function submitPlan() {
  const txt = document.getElementById('jobJson').value.trim();
  if (!txt) { alert('Paste a job JSON first'); return; }
  let job;
  try { job = JSON.parse(txt); } catch(e) { alert('Invalid JSON: '+e.message); return; }
  const dry = document.getElementById('dryRun').checked;
  const strategy = document.getElementById('strategySelect').value || 'greedy';
  try {
    await fetchJSON('/api/plan', {method:'POST', headers:{'content-type':'application/json'}, body: JSON.stringify({job: job, dry_run: dry, strategy})});
    await refresh();
  } catch(e) {
    alert('Plan failed: '+e.message);
  }
}

async function planCatalog() {
  if (!JOB_CATALOG.length) { alert('No jobs discovered in jobs/ yet.'); return; }
  const jobs = JOB_CATALOG.map(entry => entry.job).filter(j => j && typeof j === 'object');
  if (!jobs.length) { alert('Job catalog has no valid entries to plan.'); return; }
  const dry = document.getElementById('dryRun').checked;
  const strategy = document.getElementById('strategySelect').value || 'greedy';
  try {
    await fetchJSON('/api/plan_batch', {method:'POST', headers:{'content-type':'application/json'}, body: JSON.stringify({jobs, dry_run: dry, strategy})});
    await refresh();
  } catch (e) {
    alert('Batch planning failed: '+e.message);
  }
}

async function runDemo(dry=true) {
  const strategy = document.getElementById('strategySelect').value || 'greedy';
  try {
    await fetchJSON('/api/plan_demo', {method:'POST', headers:{'content-type':'application/json'}, body: JSON.stringify({dry_run: dry, strategy})});
    await refresh();
  } catch(e) {
    alert('Demo failed: '+e.message);
  }
}

async function applyObs() {
  const txt = document.getElementById('obsJson').value.trim();
  if (!txt) { alert('Paste an observation JSON'); return; }
  let payload;
  try { payload = JSON.parse(txt); } catch(e) { alert('Invalid JSON: '+e.message); return; }
  try {
    await fetchJSON('/api/observe', {method:'POST', headers:{'content-type':'application/json'}, body: JSON.stringify(payload)});
    await refresh();
  } catch(e) {
    alert('Observation failed: '+e.message);
  }
}

window.addEventListener('resize', () => {
  if (TOPO_RESIZE) clearTimeout(TOPO_RESIZE);
  TOPO_RESIZE = setTimeout(() => {
    renderTopology();
  }, 200);
});

setInterval(refresh, 2000);
CHAOS_STATUS_TIMER = setInterval(refreshChaosStatus, 5000);
window.addEventListener('load', () => {
  refresh();
  loadJobCatalog();
  loadChaosOptions();
  refreshChaosStatus();
});
</script>

</body>
</html>
"""


@app.get("/")
def index():
    resp = make_response(_INDEX_HTML.replace("__REMOTE_LABEL__", REMOTE_LABEL))
    resp.headers["Content-Type"] = "text/html; charset=utf-8"
    return resp


# ----------------- CLI entry -----------------


def main():
    ap = argparse.ArgumentParser(description="Fabric DT Dashboard")
    ap.add_argument("--host", default=os.environ.get("FABRIC_UI_HOST", "127.0.0.1"))
    ap.add_argument(
        "--port", type=int, default=int(os.environ.get("FABRIC_UI_PORT", "8090"))
    )
    ap.add_argument(
        "--remote", help="Remote Fabric DT API base URL (e.g. http://127.0.0.1:8080)"
    )
    ap.add_argument(
        "--remote-timeout",
        type=float,
        default=REMOTE_TIMEOUT,
        help="Timeout in seconds when contacting the remote API",
    )
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    if args.remote is not None or args.remote_timeout != REMOTE_TIMEOUT:
        target_remote = args.remote if args.remote is not None else REMOTE_BASE
        _configure_runtime(target_remote, args.remote_timeout)
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
