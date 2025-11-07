#!/usr/bin/env python3
"""
Digital Twin State service (DT).

- Loads ./nodes/*.yaml descriptors and sim/topology.yaml
- Maintains "effective" state by merging live overrides (sim/overrides.json)
- Exposes:
    GET  /state/nodes      -> effective nodes (after overrides)
    GET  /state/links      -> effective link DB (after overrides)
    POST /observe          -> apply patch-like overrides (as used by sim/chaos.py)
    POST /plan             -> simple greedy planner (format/accelerator aware)

Planner contract (used by sim/montecarlo.py):
    POST /plan
    {
      "jobs": { "jobs": [ { "id": "...", "stages": [...], "deadline_ms": ... } ] },
      "nodes": [optional external node list; else DT nodes],
      "topology": [optional external topo; else DT topo],
      "mode": "montecarlo" | "default"
    }
    -> { "assignments": { "<stage_id>": "<node_name>", ... } }

Run:
    pip install fastapi uvicorn pyyaml
    python3 dt/state.py --nodes nodes --topology sim/topology.yaml --overrides sim/overrides.json --port 5055
"""

from __future__ import annotations
import argparse, json, os, threading, time, math, statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import yaml
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# -----------------------------
# File IO
# -----------------------------

def load_yaml(p: Path) -> Any:
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def safe_read_json(p: Path, default: Any) -> Any:
    try:
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        pass
    return default

def write_json(p: Path, data: Any):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")

# -----------------------------
# Link helpers
# -----------------------------

def link_key(a: str, b: str) -> str:
    return "|".join(sorted([a, b]))

def build_link_db(topology: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    db: Dict[str, Dict[str, Any]] = {}
    for ln in topology.get("links", []) or []:
        a = ln.get("a"); b = ln.get("b")
        if not a or not b:
            continue
        prof = ln.get("profile")
        rec = {}
        # default inline values
        for k in ("speed_gbps","rtt_ms","jitter_ms","loss_pct","ecn","mtu_bytes"):
            if k in ln: rec[k] = ln[k]
        if prof:
            # if a profile exists, also bubble profile name so planners can resolve if needed
            rec["profile"] = prof
        rec["down"] = False
        db[link_key(a,b)] = rec
    return db

def get_link_metrics(db: Dict[str, Dict[str, Any]], a: str, b: str,
                     default_speed_gbps: float = 1.0,
                     default_rtt_ms: float = 5.0,
                     default_jitter_ms: float = 0.5) -> Tuple[float, float, float, float, bool]:
    k = link_key(a,b)
    if k in db:
        d = db[k]
        return (
            float(d.get("speed_gbps", default_speed_gbps)),
            float(d.get("rtt_ms", default_rtt_ms)),
            float(d.get("jitter_ms", default_jitter_ms)),
            float(d.get("loss_pct", 0.0)),
            bool(d.get("down", False)),
        )
    # site-wide link fallback not implemented here; callers may insert site names directly
    return default_speed_gbps, default_rtt_ms, default_jitter_ms, 0.0, False

# -----------------------------
# Cost model & planner bits
# -----------------------------

def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default

def node_compute_capacity(node: Dict[str, Any]) -> float:
    cpu = node.get("cpu", {})
    cores = safe_float(cpu.get("cores"), 0)
    ghz   = safe_float(cpu.get("base_ghz"), 0)
    derate = safe_float(node.get("_eff", {}).get("thermal_derate"), 0.0)
    if node.get("_eff", {}).get("down", False):
        return 0.0
    # power cap hint: if present, scale capacity (crude)
    pcap = safe_float(node.get("_eff", {}).get("power_cap_w"), 0.0)
    cap = max(0.0, cores * ghz * (1.0 - derate))
    if pcap > 0:
        cap *= 0.7  # simple cap effect
    return cap

def accel_multiplier(node: Dict[str, Any], stage: Dict[str, Any]) -> float:
    fmts_node = set(node.get("formats_supported") or [])
    fmt_allow = set(stage.get("allowed_formats") or [])
    fmt_dis   = set(stage.get("disallowed_formats") or [])

    mult = 1.0
    # CUDA
    if ("cuda" in fmts_node) and ("cuda" not in fmt_dis) and (not fmt_allow or "cuda" in fmt_allow):
        score = safe_float((node.get("gpu") or {}).get("accel_score"), 0.0)
        if score > 0:
            mult = max(mult, min(1.0 + score/10.0, 6.0))
    # NPU
    acc = node.get("accelerators") or {}
    if ("npu" in fmts_node) and ("npu" not in fmt_dis) and (not fmt_allow or "npu" in fmt_allow):
        tops = safe_float(acc.get("npu_tops"), 0.0)
        mult = max(mult, min(1.0 + tops/10.0, 3.0))
    # WASM/native: modest boost if stage prefers them
    if ("wasm" in fmts_node) and (fmt_allow and "wasm" in fmt_allow):
        mult = max(mult, 1.15)
    if ("native" in fmts_node) and (fmt_allow and "native" in fmt_allow):
        mult = max(mult, 1.05)
    return mult

def stage_compute_time_ms(stage: Dict[str, Any], node: Dict[str, Any]) -> float:
    cap = node_compute_capacity(node)
    if cap <= 0: return float("inf")
    size_mb = safe_float(stage.get("size_mb"), 10.0)
    cpu_req = safe_float((stage.get("resources") or {}).get("cpu_cores"), 1.0)
    base = max(20.0, size_mb * 2.0 + cpu_req * 100.0)
    base /= max(1.0, cap/10.0)
    base /= accel_multiplier(node, stage)
    # trust penalty (optional)
    trust = safe_float((node.get("labels") or {}).get("trust"), 0.9)
    if trust < 0.75:
        base *= 1.05
    return base

def transfer_time_ms(src: str, dst: str, size_mb: float, link_db: Dict[str, Dict[str, Any]]) -> float:
    if size_mb <= 0 or src == dst:
        return 0.0
    speed_gbps, rtt_ms, jitter_ms, loss_pct, down = get_link_metrics(link_db, src, dst, default_speed_gbps=1.0, default_rtt_ms=5.0)
    if down:
        return float("inf")
    mbps = speed_gbps * 1000.0
    eff_mbps = mbps * (1.0 - min(0.3, loss_pct/100.0)) * 0.85
    xfer = (size_mb * 8.0) / max(1.0, eff_mbps) * 1000.0
    return xfer + rtt_ms + jitter_ms

def greedy_place(job: Dict[str, Any], nodes: List[Dict[str, Any]], link_db: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
    assign: Dict[str, str] = {}
    stages = job.get("stages") or []
    prev_node = None
    for st in stages:
        candidates = []
        for n in nodes:
            if n.get("_eff", {}).get("down", False):
                continue
            # format gate
            allowed = set(st.get("allowed_formats") or [])
            disallowed = set(st.get("disallowed_formats") or [])
            fmts = set(n.get("formats_supported") or [])
            if allowed and not (fmts & allowed):
                continue
            if disallowed and (fmts & disallowed):
                continue
            # compute + transfer
            comp = stage_compute_time_ms(st, n)
            xfer = 0.0 if prev_node is None else transfer_time_ms(prev_node, n["name"], safe_float(st.get("size_mb"), 10.0), link_db)
            candidates.append((comp + xfer, n["name"]))
        if not candidates:
            # fallback: any alive node
            fallback = next((nn["name"] for nn in nodes if not nn.get("_eff", {}).get("down", False)), None)
            assign[st["id"]] = fallback or "unavailable"
        else:
            candidates.sort(key=lambda x: x[0])
            choice = candidates[0][1]
            assign[st["id"]] = choice
            prev_node = choice
    return assign

# -----------------------------
# DT State
# -----------------------------

class DTState:
    def __init__(self, nodes_dir: Path, topology_path: Path, overrides_path: Path, refresh_sec: float = 1.0):
        self.nodes_dir = nodes_dir
        self.topology_path = topology_path
        self.overrides_path = overrides_path
        self.refresh_sec = max(0.2, refresh_sec)
        self._lock = threading.RLock()

        self._nodes_raw: List[Dict[str, Any]] = []
        self._topology: Dict[str, Any] = {}
        self._link_db_base: Dict[str, Dict[str, Any]] = {}
        self._overrides: Dict[str, Any] = {"links": {}, "nodes": {}}

        self._nodes_eff: List[Dict[str, Any]] = []
        self._link_db_eff: Dict[str, Dict[str, Any]] = {}

        self._load_all()
        self._apply_overrides()
        self._start_watcher()

    # ---------- Loading & applying ----------

    def _load_nodes(self) -> List[Dict[str, Any]]:
        out = []
        for f in sorted(self.nodes_dir.glob("*.yaml")):
            try:
                out.append(load_yaml(f))
            except Exception:
                continue
        return out

    def _load_topology(self) -> Dict[str, Any]:
        if self.topology_path.exists():
            return load_yaml(self.topology_path)
        return {"links": []}

    def _load_overrides(self) -> Dict[str, Any]:
        return safe_read_json(self.overrides_path, {"links": {}, "nodes": {}})

    def _load_all(self):
        with self._lock:
            self._nodes_raw = self._load_nodes()
            self._topology = self._load_topology()
            self._link_db_base = build_link_db(self._topology)
            self._overrides = self._load_overrides()

    def _apply_overrides(self):
        with self._lock:
            # Effective link DB = base + overrides.links (shallow field-level merge)
            L = {k: dict(v) for k, v in self._link_db_base.items()}
            for k, patch in (self._overrides.get("links") or {}).items():
                base = L.get(k, {})
                base.update(patch)
                L[k] = base
            self._link_db_eff = L

            # Effective nodes = raw nodes with a transient _eff field merged from overrides.nodes
            N = []
            for n in self._nodes_raw:
                m = dict(n)
                o = (self._overrides.get("nodes") or {}).get(n.get("name", ""), {})
                m["_eff"] = {
                    "down": bool(o.get("down", False)),
                    "power_cap_w": o.get("power_cap_w"),
                    "thermal_derate": o.get("thermal_derate"),
                    "clock_skew_ms": o.get("clock_skew_ms"),
                    "packet_dup": o.get("packet_dup"),
                    "packet_reorder": o.get("packet_reorder"),
                }
                N.append(m)
            self._nodes_eff = N

    # ---------- Public getters ----------

    def get_nodes(self) -> List[Dict[str, Any]]:
        with self._lock:
            return self._nodes_eff

    def get_link_db(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            return self._link_db_eff

    def get_topology(self) -> Dict[str, Any]:
        with self._lock:
            return self._topology

    # ---------- Observe/override ----------

    def observe(self, action: str, payload: Dict[str, Any]):
        """
        Accepts patches like those emitted by sim/chaos.py:
          {"action":"apply","payload":{"type":"link","key":"A|B","changes":{"speed_gbps":1.0}}}
          {"action":"revert","payload":{"type":"node","node":"ws-001","fields":["down"]}}
        """
        dirty = False
        o = self._overrides
        typ = (payload or {}).get("type")
        if action == "apply":
            if typ == "link":
                k = payload.get("key")
                ch = payload.get("changes") or {}
                if k:
                    cur = o.setdefault("links", {}).get(k, {})
                    cur.update(ch)
                    o["links"][k] = cur
                    dirty = True
            elif typ == "node":
                node = payload.get("node")
                ch = payload.get("changes") or {}
                if node:
                    cur = o.setdefault("nodes", {}).get(node, {})
                    cur.update(ch)
                    o["nodes"][node] = cur
                    dirty = True
        elif action == "revert":
            if typ == "link":
                k = payload.get("key")
                fields = payload.get("fields") or []
                if k and k in o.get("links", {}):
                    for f in fields:
                        o["links"][k].pop(f, None)
                    if not o["links"][k]:
                        o["links"].pop(k, None)
                    dirty = True
            elif typ == "node":
                node = payload.get("node")
                fields = payload.get("fields") or []
                if node and node in o.get("nodes", {}):
                    for f in fields:
                        o["nodes"][node].pop(f, None)
                    if not o["nodes"][node]:
                        o["nodes"].pop(node, None)
                    dirty = True

        if dirty:
            write_json(self.overrides_path, o)
            self._apply_overrides()

    # ---------- Planning ----------

    def plan(self, jobs_payload: Dict[str, Any],
             nodes_ext: Optional[List[Dict[str, Any]]] = None,
             topology_ext: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        with self._lock:
            nodes = nodes_ext if nodes_ext else self._nodes_eff
            link_db = build_link_db(topology_ext) if topology_ext else self._link_db_eff
        jobs_list = (jobs_payload or {}).get("jobs") or []
        if not jobs_list:
            return {}

        # For MVP, if multiple jobs are provided, plan the first; or flatten by sequential planning
        # Here we plan the first job; extend if needed.
        job = jobs_list[0]
        return greedy_place(job, nodes, link_db)

    # ---------- Background watcher ----------

    def _watch_loop(self):
        last_mtime = None
        while True:
            try:
                if self.overrides_path.exists():
                    m = self.overrides_path.stat().st_mtime
                    if last_mtime is None or m > last_mtime:
                        last_mtime = m
                        self._overrides = self._load_overrides()
                        self._apply_overrides()
                # nodes/topology hot-reload (optional, every N ticks)
                # cheap: reload every ~5s
                if int(time.time()) % 5 == 0:
                    self._nodes_raw = self._load_nodes()
                    self._topology = self._load_topology()
                    self._link_db_base = build_link_db(self._topology)
                    self._apply_overrides()
            except Exception:
                pass
            time.sleep(self.refresh_sec)

    def _start_watcher(self):
        t = threading.Thread(target=self._watch_loop, daemon=True)
        t.start()

    # expose paths for watcher
    @property
    def overrides_path(self) -> Path:
        return self._overrides_path
    @overrides_path.setter
    def overrides_path(self, p: Path):
        self._overrides_path = p

# -----------------------------
# FastAPI wiring
# -----------------------------

def make_app(state: DTState) -> FastAPI:
    app = FastAPI(title="NeoCloud DT State", version="0.1.0")

    @app.get("/state/nodes")
    def get_nodes():
        return JSONResponse(state.get_nodes())

    @app.get("/state/links")
    def get_links():
        return JSONResponse(state.get_link_db())

    @app.post("/observe")
    async def post_observe(req: Request):
        """
        Body: {"action": "apply"|"revert", "payload": {...}}
        Matches sim/chaos.py posting style.
        """
        body = await req.json()
        action = body.get("action")
        payload = body.get("payload") or {}
        if action not in ("apply","revert"):
            return JSONResponse({"ok": False, "error": "invalid action"}, status_code=400)
        state.observe(action, payload)
        return JSONResponse({"ok": True})

    @app.post("/plan")
    async def post_plan(req: Request):
        """
        Accepts jobs/nodes/topology (optional) and returns assignments.
        """
        body = await req.json()
        jobs = body.get("jobs") or {}
        nodes = body.get("nodes")
        topo  = body.get("topology")
        assign = state.plan(jobs, nodes_ext=nodes, topology_ext=topo)
        return JSONResponse({"assignments": assign})

    return app

# -----------------------------
# CLI / entry
# -----------------------------

def main():
    ap = argparse.ArgumentParser(description="DT State service")
    ap.add_argument("--nodes", default="nodes", help="Directory with node YAMLs")
    ap.add_argument("--topology", default="sim/topology.yaml", help="Topology YAML path")
    ap.add_argument("--overrides", default="sim/overrides.json", help="Overrides JSON path")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5055)
    ap.add_argument("--refresh", type=float, default=1.0, help="Refresh period (s) for file watcher")
    ap.add_argument("--no-server", action="store_true", help="Just build state once and print a summary")
    args = ap.parse_args()

    nodes_dir = Path(args.nodes)
    topo_path = Path(args.topology)
    ovr_path  = Path(args.overrides)

    state = DTState(nodes_dir=nodes_dir, topology_path=topo_path, overrides_path=ovr_path, refresh_sec=args.refresh)

    if args.no_server:
        ns = state.get_nodes()
        ls = state.get_link_db()
        print(f"[dt] nodes={len(ns)} links={len(ls)}")
        # tiny preview
        if ns:
            n0 = ns[0]
            print("[dt] example node:", n0.get("name"), "cap=", round(node_compute_capacity(n0), 3))
        return 0

    # lazy import uvicorn only when serving
    try:
        import uvicorn
    except Exception as e:
        print("Error: uvicorn not installed. `pip install uvicorn`")
        return 2

    app = make_app(state)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

