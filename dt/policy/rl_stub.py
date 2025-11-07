#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dt/policy/rl_stub.py — Minimal RL scaffold for node selection.

Purpose
-------
Provide a light, swappable RL component you can:
- call to pick a node (ε-greedy)
- update with observed reward (TD(0))
- persist between runs

It does NOT run an environment loop itself; your simulator or real execution
should call `record_transition(...)` once you know the outcome latency/energy/risk.

API
---
rl = RLPolicy(persist_path="sim/rl_state.json",
              algo="qlearning",        # or "sarsa"
              epsilon=0.15, alpha=0.3, gamma=0.92)

node = rl.choose_node(stage, candidates)   # candidates: {name: node_obj}
# ... run/simulate ...
rl.record_transition(stage, prev_node, node, reward, next_candidates, next_node=None)

If you prefer blending with a heuristic score:
bonuses = rl.score_candidates(stage, candidates)  # dict: node -> bonus_ms (negative is better)

Reward
------
We normalize to [0,1] where higher is better. A quick mapping:
    r = 1 - min(1, compute_ms / scale_ms)
Optionally blend energy/risk.

Persistence
-----------
JSON file: sim/rl_state.json storing { "(sig@@node)": value }.

"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List
import json
import math
import random
import time

def safe_float(x: Any, d: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return d

def _bucket(x: float, step: float, lo: float = 0.0, hi: float = 999999.0) -> int:
    x = max(lo, min(hi, x))
    return int(x // step)

def _stage_sig(stage: Dict[str, Any]) -> str:
    typ = stage.get("type") or "generic"
    res = stage.get("resources") or {}
    size_mb = safe_float(stage.get("size_mb"), 0.0)
    cpu = safe_float(res.get("cpu_cores"), 0.0)
    mem = safe_float(res.get("mem_gb"), 0.0)
    vram = safe_float(res.get("gpu_vram_gb"), 0.0)
    b_size = _bucket(size_mb, 50.0, 0.0, 500.0)
    b_cpu  = _bucket(cpu, 1.0, 0.0, 16.0)
    b_mem  = _bucket(mem, 2.0, 0.0, 64.0)
    b_vram = _bucket(vram, 1.0, 0.0, 24.0)
    return f"{typ}|s{b_size}|c{b_cpu}|m{b_mem}|g{b_vram}"

@dataclass
class RLConfig:
    epsilon: float = 0.15   # exploration prob
    alpha: float = 0.3      # learning rate
    gamma: float = 0.92     # discount
    algo: str = "qlearning" # or "sarsa"
    reward_scale_ms: float = 2000.0
    energy_weight: float = 0.0
    risk_weight: float = 0.0

class RLPolicy:
    def __init__(self, persist_path: Optional[str] = "sim/rl_state.json", **kwargs):
        self.cfg = RLConfig(**{**RLConfig().__dict__, **kwargs})
        self.persist_path = Path(persist_path) if persist_path else None
        self.Q: Dict[str, float] = {}
        if self.persist_path and self.persist_path.exists():
            try:
                self.Q = json.loads(self.persist_path.read_text(encoding="utf-8"))
            except Exception:
                self.Q = {}

    # ---------- helpers ----------
    def _key(self, stage_sig: str, node_name: str) -> str:
        return f"{stage_sig}@@{node_name}"

    def value(self, stage_sig: str, node_name: str) -> float:
        return float(self.Q.get(self._key(stage_sig, node_name), 0.0))

    def set_value(self, stage_sig: str, node_name: str, v: float):
        self.Q[self._key(stage_sig, node_name)] = float(v)

    def save(self):
        if not self.persist_path:
            return
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        self.persist_path.write_text(json.dumps(self.Q, indent=2), encoding="utf-8")

    # ---------- action selection ----------
    def choose_node(self, stage: Dict[str, Any], candidates: Dict[str, Dict[str, Any]]) -> str:
        sig = _stage_sig(stage)
        names = list(candidates.keys())
        if not names:
            return ""
        # ε-greedy
        if random.random() < self.cfg.epsilon:
            return random.choice(names)
        # exploit
        best = None
        best_v = -1e9
        for n in names:
            v = self.value(sig, n)
            if v > best_v:
                best_v = v
                best = n
        return best or random.choice(names)

    def score_candidates(self, stage: Dict[str, Any], candidates: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """Return a *bonus* score you can subtract from greedy latency (negative is better)."""
        sig = _stage_sig(stage)
        out = {}
        # Map Q in [min, max] to a ms-bonus range (e.g., up to 10ms)
        vals = [self.value(sig, n) for n in candidates.keys()]
        if not vals:
            return {n: 0.0 for n in candidates.keys()}
        vmin, vmax = min(vals), max(vals)
        span = max(1e-6, vmax - vmin)
        for n in candidates.keys():
            v = self.value(sig, n)
            norm = (v - vmin) / span  # 0..1
            bonus_ms = -10.0 * norm   # better Q → more negative → preferred
            out[n] = bonus_ms
        return out

    # ---------- learning ----------
    def _reward(self, compute_ms: float, energy_kj: Optional[float], risk: Optional[float]) -> float:
        # Normalize to [0,1], higher is better
        r_lat = 1.0 - min(1.0, max(0.0, compute_ms / max(1.0, self.cfg.reward_scale_ms)))
        e = energy_kj if (energy_kj is not None) else 0.0
        r = risk if (risk is not None) else 0.0
        # Simple penalties normalized to 0..1 (heuristic)
        e_norm = 1.0 - min(1.0, e / 5.0)     # 0kJ -> 1 ; 5kJ+ -> 0
        r_norm = 1.0 - min(1.0, r)           # 0 -> 1 ; 1 -> 0
        mix = (1.0 - self.cfg.energy_weight - self.cfg.risk_weight)
        mix = max(0.0, mix)
        total = mix * r_lat + self.cfg.energy_weight * e_norm + self.cfg.risk_weight * r_norm
        return float(total)

    def record_transition(
        self,
        stage: Dict[str, Any],
        prev_node: Optional[str],
        node: str,
        compute_ms: float,
        energy_kj: Optional[float] = None,
        risk: Optional[float] = None,
        next_stage: Optional[Dict[str, Any]] = None,
        next_candidates: Optional[Dict[str, Dict[str, Any]]] = None,
        next_node: Optional[str] = None,
    ):
        """
        One TD(0) update. If next_stage/next_candidates is provided:
          - SARSA uses Q(next_sig,next_node) (if next_node not given, it will pick one ε-greedily)
          - Q-learning uses max_a' Q(next_sig,a')
        """
        if not node:
            return
        sig = _stage_sig(stage)
        s_key = self._key(sig, node)
        q = float(self.Q.get(s_key, 0.0))
        rwd = self._reward(compute_ms, energy_kj, risk)

        target = rwd
        if next_stage and next_candidates:
            next_sig = _stage_sig(next_stage)
            if self.cfg.algo.lower().startswith("sarsa"):
                # on-policy: use the next chosen action
                if not next_node:
                    # pick using current policy (ε-greedy)
                    next_node = self.choose_node(next_stage, next_candidates)
                q_next = float(self.Q.get(self._key(next_sig, next_node), 0.0))
            else:
                # Q-learning: max over actions
                q_next = 0.0
                for n2 in next_candidates.keys():
                    q_next = max(q_next, float(self.Q.get(self._key(next_sig, n2), 0.0)))
            target = rwd + self.cfg.gamma * q_next

        # TD(0) update
        new_q = q + self.cfg.alpha * (target - q)
        self.Q[s_key] = float(new_q)

    # Convenience to hook into your Greedy planner:
    def bonus_ms_for(self, stage: Dict[str, Any], node_name: str) -> float:
        """Return a small negative number for nodes with higher Q."""
        scores = self.score_candidates(stage, {node_name: {}})
        return float(scores.get(node_name, 0.0))

