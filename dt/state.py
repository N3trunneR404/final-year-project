#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dt/state.py — Digital Twin runtime state for the Fabric simulator.

Responsibilities
---------------
- Load per-node descriptors:          ./nodes/*.yaml
- Load optional topology:             ./sim/topology.yaml  (links, defaults)
- Watch & merge runtime overrides:    ./sim/overrides.json (written by sim/chaos.py)
- Maintain thread-safe resource view: capacities, reservations, queues
- Offer a compact API for dt/api.py:
    • snapshot()                 → dict (nodes, links, ts)
    • reserve(req)               → reservation_id or None
    • release(reservation_id)    → bool
    • score_node_basic(stage,n)  → float (lower is better)
    • apply_observation(payload) → merge ad-hoc updates (used by /observe)

Design notes
------------
- No hard dependency on Flask here (pure state). dt/api.py can import and call it.
- Only non-stdlib dep is PyYAML. (requests is optional if you later push updates out.)
- Links are stored as an undirected map keyed by "A|B".
- Dynamic/ephemeral state is kept under node["dyn"] and link["dyn"].

Paths (configurable via constructor)
-----------------------------------
nodes_dir        default: "nodes"
topology_path    default: "sim/topology.yaml" (optional)
overrides_path   default: "sim/overrides.json" (optional)

"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


# ----------------------------- helpers -----------------------------

def link_key(a: str, b: str) -> str:
    return "|".join(sorted([a, b]))


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def utc_ms() -> int:
    return int(time.time() * 1000)


# ----------------------------- data classes -----------------------------

@dataclass
class NodeDyn:
    """Mutable, runtime-only fields for a node."""
    down: bool = False
    thermal_derate: float = 0.0         # 0..1
    power_cap_w: Optional[float] = None
    clock_skew_ms: Optional[float] = None
    packet_dup: Optional[float] = None
    packet_reorder: Optional[float] = None
    used_cpu_cores: float = 0.0
    used_mem_gb: float = 0.0
    used_gpu_vram_gb: float = 0.0
    reservations: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # res_id -> req


@dataclass
class LinkDyn:
    """Mutable, runtime-only fields for a link."""
    down: bool = False
    speed_gbps: Optional[float] = None
    rtt_ms: Optional[float] = None
    jitter_ms: Optional[float] = None
    loss_pct: Optional[float] = None
    ecn: Optional[bool] = None


# ----------------------------- DT State -----------------------------

class DTState:
    def __init__(
        self,
        nodes_dir: str = "nodes",
        topology_path: str = "sim/topology.yaml",
        overrides_path: str = "sim/overrides.json",
        watch_interval_sec: float = 0.5,
        auto_start_watchers: bool = True,
    ):
        self.nodes_dir = Path(nodes_dir)
        self.topology_path = Path(topology_path)
        self.overrides_path = Path(overrides_path)

        self._lock = threading.RLock()

        # Static-ish structures
        self.nodes_by_name: Dict[str, Dict[str, Any]] = {}  # includes 'dyn'
        self.links_by_key: Dict[str, Dict[str, Any]] = {}   # includes 'dyn'
        self.defaults: Dict[str, Any] = {}

        # Overrides (raw copies of sim/overrides.json)
        self._overrides: Dict[str, Any] = {"nodes": {}, "links": {}}
        self._overrides_mtime: float = 0.0

        # Node/Topology mtimes to allow hot reloads if you want to extend it
        self._nodes_mtime: float = 0.0
        self._topology_mtime: float = 0.0

        # Reservation counter
        self._res_seq: int = 1

        # Initial load
        self._load_nodes_locked()
        self._load_topology_locked()
        self._load_overrides_locked(apply_now=True)

        # Background watcher for overrides (and optionally hot-reload topology)
        self._watch_interval = max(0.2, float(watch_interval_sec))
        self._watch_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        if auto_start_watchers:
            self.start()

    # -------- public lifecycle --------

    def start(self):
        if self._watch_thread and self._watch_thread.is_alive():
            return
        self._stop_event.clear()
        self._watch_thread = threading.Thread(target=self._watch_loop, name="DTStateWatch", daemon=True)
        self._watch_thread.start()

    def stop(self):
        self._stop_event.set()
        if self._watch_thread:
            self._watch_thread.join(timeout=2.0)

    # -------- loads & merges --------

    def _load_nodes_locked(self):
        """Load ./nodes/*.yaml into nodes_by_name with fresh dyn slots."""
        with self._lock:
            nodes: Dict[str, Dict[str, Any]] = {}
            latest_mtime = self._nodes_mtime
            for f in sorted(self.nodes_dir.glob("*.yaml")):
                try:
                    stat = f.stat()
                    latest_mtime = max(latest_mtime, stat.st_mtime)
                    data = yaml.safe_load(f.read_text(encoding="utf-8"))
                    name = data.get("name")
                    if not name:
                        continue
                    # Ensure dyn exists and keep capacity-derived caches
                    data.setdefault("dyn", NodeDyn().__dict__.copy())
                    # Cached capacities
                    self._compute_and_cache_capacities(data)
                    nodes[name] = data
                except Exception as e:
                    print(f"[state] WARN: failed to load node {f.name}: {e}")

            self.nodes_by_name = nodes
            self._nodes_mtime = latest_mtime

    def _load_topology_locked(self):
        """Load topology (links + defaults) if present."""
        with self._lock:
            if not self.topology_path.exists():
                self.links_by_key = {}
                self.defaults = {}
                return
            try:
                stat = self.topology_path.stat()
                topo = yaml.safe_load(self.topology_path.read_text(encoding="utf-8"))
                self._topology_mtime = stat.st_mtime

                # Defaults (optional; used if you want to fall back)
                self.defaults = topo.get("defaults", {}) or {}

                links: Dict[str, Dict[str, Any]] = {}
                for ln in (topo.get("links") or []):
                    a, b = ln.get("a"), ln.get("b")
                    if not a or not b:
                        continue
                    k = link_key(a, b)
                    lnd = {
                        "a": a, "b": b,
                        "profile": ln.get("profile"),
                        "qos_class": ln.get("qos_class"),
                        "scope": ln.get("scope", "site"),
                        "subnet": ln.get("subnet"),
                        "base": {
                            # Allow explicit metrics in link inline
                            "speed_gbps": ln.get("speed_gbps"),
                            "rtt_ms": ln.get("rtt_ms"),
                            "jitter_ms": ln.get("jitter_ms"),
                            "loss_pct": ln.get("loss_pct"),
                            "ecn": ln.get("ecn"),
                        },
                        "dyn": LinkDyn().__dict__.copy(),
                    }
                    # Strip Nones from base for cleanliness
                    lnd["base"] = {k2: v2 for k2, v2 in lnd["base"].items() if v2 is not None}
                    links[k] = lnd
                self.links_by_key = links
            except Exception as e:
                print(f"[state] WARN: failed to load topology: {e}")
                self.links_by_key = {}
                self.defaults = {}

    def _load_overrides_locked(self, apply_now: bool = True):
        """Load sim/overrides.json if present; optionally apply immediately."""
        with self._lock:
            if not self.overrides_path.exists():
                self._overrides = {"nodes": {}, "links": {}}
                self._overrides_mtime = 0.0
                return
            try:
                stat = self.overrides_path.stat()
                if stat.st_mtime <= self._overrides_mtime:
                    return
                raw = json.loads(self.overrides_path.read_text(encoding="utf-8"))
                self._overrides = {
                    "nodes": raw.get("nodes", {}) or {},
                    "links": raw.get("links", {}) or {},
                }
                self._overrides_mtime = stat.st_mtime
                if apply_now:
                    self._apply_overrides_locked()
            except Exception as e:
                print(f"[state] WARN: failed to load overrides.json: {e}")

    def _apply_overrides_locked(self):
        """Merge self._overrides into node/link dyn fields."""
        # Nodes
        for nname, changes in self._overrides.get("nodes", {}).items():
            n = self.nodes_by_name.get(nname)
            if not n:
                continue
            dyn = n.setdefault("dyn", NodeDyn().__dict__.copy())
            # Only accept known fields
            for k in ("down", "power_cap_w", "thermal_derate", "clock_skew_ms",
                      "packet_dup", "packet_reorder"):
                if k in changes:
                    dyn[k] = changes[k]

        # Links
        for k, changes in self._overrides.get("links", {}).items():
            l = self.links_by_key.get(k)
            if not l:
                # Permit ad-hoc links (e.g., node↔node Wi-Fi), create shell
                parts = k.split("|", 1)
                if len(parts) == 2:
                    l = {"a": parts[0], "b": parts[1], "base": {}, "dyn": LinkDyn().__dict__.copy()}
                    self.links_by_key[k] = l
                else:
                    continue
            dyn = l.setdefault("dyn", LinkDyn().__dict__.copy())
            for kk in ("down", "speed_gbps", "rtt_ms", "jitter_ms", "loss_pct", "ecn"):
                if kk in changes:
                    dyn[kk] = changes[kk]

    def _compute_and_cache_capacities(self, node: Dict[str, Any]):
        """Precompute static capacities and store under node['caps']."""
        cpu = node.get("cpu", {}) or {}
        mem = node.get("memory", {}) or {}
        gpu = node.get("gpu", {}) or {}

        cores = safe_float(cpu.get("cores"), 0.0)
        base_ghz = safe_float(cpu.get("base_ghz"), 0.0)
        ram_gb = safe_float(mem.get("ram_gb"), 0.0)
        vram_gb = safe_float(gpu.get("vram_gb"), 0.0)

        # naive "capacity units"
        cpu_units = cores * base_ghz
        node["caps"] = {
            "cpu_units": cpu_units,
            "max_cpu_cores": cores,
            "ram_gb": ram_gb,
            "gpu_vram_gb": vram_gb,
        }

    # -------- watcher loop --------

    def _watch_loop(self):
        while not self._stop_event.is_set():
            try:
                # Overrides
                self._load_overrides_locked(apply_now=True)

                # (Optional) Hot-reload topology if changed on disk
                if self.topology_path.exists():
                    stat = self.topology_path.stat()
                    if stat.st_mtime > self._topology_mtime:
                        self._load_topology_locked()

                # (Optional) Hot-reload nodes if you regenerate them
                # (commented by default to avoid flicker)
                # latest_nodes_mtime = max([f.stat().st_mtime for f in self.nodes_dir.glob("*.yaml")] + [0.0])
                # if latest_nodes_mtime > self._nodes_mtime:
                #     self._load_nodes_locked()
            except Exception as e:
                print(f"[state] WARN: watcher iteration failed: {e}")

            self._stop_event.wait(self._watch_interval)

    # -------- public API (read) --------

    def snapshot(self) -> Dict[str, Any]:
        """Return a thread-safe snapshot for UI/clients."""
        with self._lock:
            nodes = []
            for n in self.nodes_by_name.values():
                dyn = n.get("dyn", {})
                caps = n.get("caps", {})
                eff = self._effective_caps(n)
                nodes.append({
                    "name": n.get("name"),
                    "class": n.get("class"),
                    "arch": n.get("arch"),
                    "formats_supported": n.get("formats_supported", []),
                    "labels": n.get("labels", {}),
                    "network": n.get("network", {}),
                    "gpu": n.get("gpu", {}),
                    "caps": caps,
                    "dyn": dyn,
                    "effective": eff,   # remaining capacities after derates+reservations
                })

            links = []
            for k, l in self.links_by_key.items():
                links.append({
                    "key": k,
                    "a": l.get("a"),
                    "b": l.get("b"),
                    "base": l.get("base", {}),
                    "dyn": l.get("dyn", {}),
                    "effective": self._effective_link(l),
                })

            return {
                "ts": utc_ms(),
                "nodes": nodes,
                "links": links,
            }

    def get_node(self, name: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self.nodes_by_name.get(name)

    # -------- public API (write/update) --------

    def apply_observation(self, payload: Dict[str, Any]) -> None:
        """
        Merge an observation (same shape chaos uses):
        { "action": "apply"|"revert", "payload": {"type": "node"|"link", ...}}
        """
        with self._lock:
            p = payload.get("payload", {})
            typ = p.get("type")
            if typ == "node":
                node = p.get("node")
                changes = p.get("changes") or {}
                target = self.nodes_by_name.get(node)
                if not target:
                    return
                dyn = target.setdefault("dyn", NodeDyn().__dict__.copy())
                for k, v in changes.items():
                    if k in dyn:
                        dyn[k] = v
            elif typ == "link":
                k = p.get("key")
                changes = p.get("changes") or {}
                link = self.links_by_key.get(k)
                if not link:
                    # Create on the fly if key is valid
                    parts = k.split("|", 1)
                    if len(parts) == 2:
                        link = {"a": parts[0], "b": parts[1], "base": {}, "dyn": LinkDyn().__dict__.copy()}
                        self.links_by_key[k] = link
                    else:
                        return
                dyn = link.setdefault("dyn", LinkDyn().__dict__.copy())
                for kk, vv in changes.items():
                    if kk in dyn:
                        dyn[kk] = vv

    # -------- reservations --------

    def reserve(self, req: Dict[str, Any]) -> Optional[str]:
        """
        Try to reserve resources on a specific node or choose one automatically.

        req example:
        {
          "node": "ws-001",                # optional; if omitted, caller should choose a node via planner
          "cpu_cores": 2.0,
          "mem_gb": 4.0,
          "gpu_vram_gb": 2.0
        }
        """
        with self._lock:
            node_name = req.get("node")
            if not node_name:
                return None
            n = self.nodes_by_name.get(node_name)
            if not n:
                return None
            if self._is_down(n):
                return None

            eff = self._effective_caps(n)
            need_cpu = safe_float(req.get("cpu_cores"), 0.0)
            need_mem = safe_float(req.get("mem_gb"), 0.0)
            need_vram = safe_float(req.get("gpu_vram_gb"), 0.0)

            if eff["free_cpu_cores"] + 1e-9 < need_cpu:
                return None
            if eff["free_mem_gb"] + 1e-9 < need_mem:
                return None
            if eff["free_gpu_vram_gb"] + 1e-9 < need_vram:
                return None

            # allocate
            dyn = n.setdefault("dyn", NodeDyn().__dict__.copy())
            dyn["used_cpu_cores"] += need_cpu
            dyn["used_mem_gb"] += need_mem
            dyn["used_gpu_vram_gb"] += need_vram

            rid = f"res-{self._res_seq:07d}"
            self._res_seq += 1
            dyn.setdefault("reservations", {})[rid] = {
                "cpu_cores": need_cpu,
                "mem_gb": need_mem,
                "gpu_vram_gb": need_vram,
                "ts": utc_ms(),
            }
            return rid

    def release(self, node_name: str, reservation_id: str) -> bool:
        with self._lock:
            n = self.nodes_by_name.get(node_name)
            if not n:
                return False
            dyn = n.get("dyn") or {}
            res = (dyn.get("reservations") or {}).pop(reservation_id, None)
            if not res:
                return False
            dyn["used_cpu_cores"] = max(0.0, dyn.get("used_cpu_cores", 0.0) - safe_float(res.get("cpu_cores"), 0.0))
            dyn["used_mem_gb"] = max(0.0, dyn.get("used_mem_gb", 0.0) - safe_float(res.get("mem_gb"), 0.0))
            dyn["used_gpu_vram_gb"] = max(0.0, dyn.get("used_gpu_vram_gb", 0.0) - safe_float(res.get("gpu_vram_gb"), 0.0))
            return True

    # -------- scoring utility (baseline) --------

    def score_node_basic(self, stage: Dict[str, Any], node: Dict[str, Any]) -> float:
        """
        Lower is better. Very simple latency proxy:
        - Penalize 'down', thermal_derate, low CPU_units
        - Give a boost if formats_supported matches stage's allowed_formats (cuda/npu)
        """
        if self._is_down(node):
            return 1e12

        caps = node.get("caps", {})
        dyn = node.get("dyn", {})
        cpu_units = safe_float(caps.get("cpu_units"), 0.0)
        derate = safe_float(dyn.get("thermal_derate"), 0.0)

        # format preference
        allowed = set(stage.get("allowed_formats") or [])
        fmts = set(node.get("formats_supported") or [])
        fmt_bonus = 0.0
        if allowed:
            if fmts & allowed:
                fmt_bonus = -0.15  # reduce score (better)
            else:
                fmt_bonus = +0.25  # increase score (worse)

        score = (1.0 / max(1e-6, cpu_units)) * (1.0 + derate) * (1.0 + fmt_bonus)
        return max(0.0, score)

    # -------- effective capacities/links --------

    def _is_down(self, node: Dict[str, Any]) -> bool:
        dyn = node.get("dyn") or {}
        return bool(dyn.get("down", False))

    def _effective_caps(self, node: Dict[str, Any]) -> Dict[str, float]:
        caps = node.get("caps", {})
        dyn = node.get("dyn", {})
        derate = safe_float(dyn.get("thermal_derate"), 0.0)

        max_cpu = safe_float(caps.get("max_cpu_cores"), 0.0)
        max_mem = safe_float(caps.get("ram_gb"), 0.0)
        max_vram = safe_float(caps.get("gpu_vram_gb"), 0.0)

        # Thermal derate reduces effective usable CPU (you can make this fancier later)
        eff_cpu = max_cpu * (1.0 - max(0.0, min(1.0, derate)))

        used_cpu = safe_float(dyn.get("used_cpu_cores"), 0.0)
        used_mem = safe_float(dyn.get("used_mem_gb"), 0.0)
        used_vram = safe_float(dyn.get("used_gpu_vram_gb"), 0.0)

        return {
            "max_cpu_cores": max_cpu,
            "max_mem_gb": max_mem,
            "max_gpu_vram_gb": max_vram,
            "free_cpu_cores": max(0.0, eff_cpu - used_cpu),
            "free_mem_gb": max(0.0, max_mem - used_mem),
            "free_gpu_vram_gb": max(0.0, max_vram - used_vram),
        }

    def _effective_link(self, link: Dict[str, Any]) -> Dict[str, Any]:
        base = link.get("base", {}) or {}
        dyn = link.get("dyn", {}) or {}

        # Choose dyn override if set, else base, else topology defaults
        def pick(key: str, default_key: Optional[str] = None, default_val: Optional[Any] = None):
            if key in dyn and dyn[key] is not None:
                return dyn[key]
            if key in base and base[key] is not None:
                return base[key]
            if default_key:
                # Look up defaults.network
                netdef = (self.defaults.get("network") or {})
                return netdef.get(default_key, default_val)
            return default_val

        eff = {
            "down": bool(dyn.get("down", False)),
            "speed_gbps": safe_float(pick("speed_gbps", "speed_gbps", 1.0), 1.0),
            "rtt_ms": safe_float(pick("rtt_ms", "rtt_ms", 5.0), 5.0),
            "jitter_ms": safe_float(pick("jitter_ms", "jitter_ms", 0.5), 0.5),
            "loss_pct": safe_float(pick("loss_pct", "loss_pct", 0.0), 0.0),
            "ecn": bool(pick("ecn", "ecn", False)),
        }
        return eff

    # -------- disk persistence for overrides (optional) --------

    def write_overrides(self) -> None:
        """Persist current dyn states to sim/overrides.json (lossy for unknown fields)."""
        with self._lock:
            out = {"nodes": {}, "links": {}}
            for name, n in self.nodes_by_name.items():
                dyn = n.get("dyn") or {}
                # Only write meaningful keys
                nd = {}
                for k in ("down", "power_cap_w", "thermal_derate", "clock_skew_ms",
                          "packet_dup", "packet_reorder"):
                    if k in dyn and dyn[k] not in (None, False, 0, 0.0):
                        nd[k] = dyn[k]
                if nd:
                    out["nodes"][name] = nd

            for k, l in self.links_by_key.items():
                dyn = l.get("dyn") or {}
                ld = {}
                for kk in ("down", "speed_gbps", "rtt_ms", "jitter_ms", "loss_pct", "ecn"):
                    if kk in dyn and dyn[kk] not in (None, False, 0, 0.0):
                        ld[kk] = dyn[kk]
                if ld:
                    out["links"][k] = ld

            try:
                self.overrides_path.parent.mkdir(parents=True, exist_ok=True)
                self.overrides_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
                self._overrides = out
                self._overrides_mtime = self.overrides_path.stat().st_mtime
            except Exception as e:
                print(f"[state] WARN: failed to write overrides: {e}")


# ----------------------------- manual test -----------------------------

if __name__ == "__main__":
    st = DTState(auto_start_watchers=False)  # don't spawn watcher for a one-off test
    snap = st.snapshot()
    print(f"Loaded nodes: {len(snap['nodes'])}, links: {len(snap['links'])}")

    # Reserve a tiny slice on the first node (if any)
    if snap["nodes"]:
        n0 = snap["nodes"][0]["name"]
        rid = st.reserve({"node": n0, "cpu_cores": 1, "mem_gb": 2})
        print("Reservation:", n0, rid)
        if rid:
            st.release(n0, rid)
            print("Released:", rid)

