#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dt/policy/bandit.py — Online format selector (native/wasm/cuda/npu/...) via a multi-armed bandit.

Goal
----
Given a (stage, node), pick an execution *format* that minimizes runtime cost.
We treat each format as an "arm" and learn from observed outcomes.

Key design points
-----------------
- Works with CostModel but does not depend on it (planner computes a format->score, runs
  the chosen one, then reports the observed compute_ms here).
- UCB1 by default; Thompson Sampling (beta) stub included.
- State keying: (stage_signature, node_name) where stage_signature is a compact hash-ish
  of the stage shape: type + allowed/disallowed + resource tiers.

Public API
----------
bp = BanditPolicy(persist_path="sim/bandit_state.json")
fmt_list = bp.suggest_formats(stage, node)  # ranked best→worst
fmt = bp.choose_format(stage, node)         # single best
bp.record_outcome(stage, node, fmt, compute_ms, energy_kj=None, risk=None)
bp.save() / bp.load()

Integration tips
----------------
- In your planner, for each feasible node, call `bp.choose_format(stage,node)` and pass it
  to your cost evaluation (e.g., set stage.allowed_formats = [chosen] temporarily or
  pass an override to CostModel). After the run (or simulation step), call `record_outcome`.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import math
import time

# --------- small utils ---------

def safe_float(x: Any, d: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return d

def _now_ms() -> int:
    return int(time.time() * 1000)

def _stage_signature(stage: Dict[str, Any]) -> str:
    """Compact signature to cluster learning by 'similar' stages."""
    t = stage.get("type") or "generic"
    res = stage.get("resources") or {}
    fmts = sorted(set(stage.get("allowed_formats") or []))
    dis  = sorted(set(stage.get("disallowed_formats") or []))
    size = safe_float(stage.get("size_mb"), 0.0)
    c = safe_float(res.get("cpu_cores"), 0.0)
    m = safe_float(res.get("mem_gb"), 0.0)
    g = safe_float(res.get("gpu_vram_gb"), 0.0)
    # Bucketize to reduce state explosion
    size_b = int(max(0, min(9, size // 50)))   # 0..9
    c_b = int(max(0, min(8, c)))               # 0..8
    m_b = int(max(0, min(8, m // 2)))          # 0..8
    g_b = int(max(0, min(8, g)))               # 0..8
    return f"{t}|s{size_b}|c{c_b}|m{m_b}|g{g_b}|a:{','.join(fmts)}|d:{','.join(dis)}"

def _available_formats(stage: Dict[str, Any], node: Dict[str, Any]) -> List[str]:
    allowed = set(stage.get("allowed_formats") or [])
    disallowed = set(stage.get("disallowed_formats") or [])
    node_fmts = set(node.get("formats_supported") or [])
    if allowed:
        arms = list((allowed & node_fmts) - disallowed)
        if arms:
            return sorted(arms)
    # If no specific allowed, use node's formats minus disallowed
    arms = list(node_fmts - disallowed)
    if arms:
        return sorted(arms)
    # Fallback (assume native exists)
    return ["native"]

# --------- data model ---------

@dataclass
class ArmStats:
    pulls: int = 0
    avg_reward: float = 0.0   # running mean
    last_ms: Optional[float] = None
    last_energy_kj: Optional[float] = None
    last_risk: Optional[float] = None
    last_ts: Optional[int] = None

    def update(self, reward: float, compute_ms: float, energy_kj: Optional[float], risk: Optional[float]):
        self.pulls += 1
        # online mean
        self.avg_reward += (reward - self.avg_reward) / max(1, self.pulls)
        self.last_ms = compute_ms
        self.last_energy_kj = energy_kj
        self.last_risk = risk
        self.last_ts = _now_ms()

@dataclass
class ContextStats:
    # arm_name -> ArmStats
    arms: Dict[str, ArmStats]
    total_pulls: int = 0

# --------- policy ---------

class BanditPolicy:
    def __init__(self, persist_path: Optional[str] = "sim/bandit_state.json", algo: str = "ucb1"):
        self._state: Dict[str, Dict[str, Any]] = {}  # key: f"{sig}@@{node_name}"
        self.persist_path = Path(persist_path) if persist_path else None
        self.algo = algo.lower().strip()
        if self.persist_path and self.persist_path.exists():
            self.load()

    # ---- persistence ----
    def save(self):
        if not self.persist_path:
            return
        out = {}
        for k, v in self._state.items():
            # Convert ArmStats to dict
            arms = {a: asdict(st) for a, st in v["arms"].items()}
            out[k] = {"arms": arms, "total_pulls": v.get("total_pulls", 0)}
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        self.persist_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    def load(self):
        if not self.persist_path or not self.persist_path.exists():
            return
        try:
            raw = json.loads(self.persist_path.read_text(encoding="utf-8"))
            state: Dict[str, Dict[str, Any]] = {}
            for k, v in raw.items():
                arms_d = {}
                for a, st in (v.get("arms") or {}).items():
                    arms_d[a] = ArmStats(**st)
                state[k] = {"arms": arms_d, "total_pulls": int(v.get("total_pulls", 0))}
            self._state = state
        except Exception as e:
            print(f"[bandit] WARN: failed to load state: {e}")

    # ---- core selection ----
    def _ctx_key(self, stage: Dict[str, Any], node: Dict[str, Any]) -> str:
        return f"{_stage_signature(stage)}@@{node.get('name')}"

    def _ensure_ctx(self, arms: List[str], key: str) -> ContextStats:
        ctx = self._state.get(key)
        if not ctx:
            ctx = {"arms": {}, "total_pulls": 0}
            self._state[key] = ctx
        for a in arms:
            if a not in ctx["arms"]:
                ctx["arms"][a] = ArmStats()
        # prune unknown arms (schema changes)
        for a in list(ctx["arms"].keys()):
            if a not in arms:
                ctx["arms"].pop(a, None)
        return ContextStats(arms=ctx["arms"], total_pulls=int(ctx.get("total_pulls", 0)))

    def _ucb1_scores(self, ctx: ContextStats) -> Dict[str, float]:
        # UCB1: score = avg + sqrt(2 ln N / n)
        N = max(1, ctx.total_pulls)
        scores = {}
        for a, st in ctx.arms.items():
            if st.pulls == 0:
                scores[a] = float("+inf")
            else:
                bonus = math.sqrt(2.0 * math.log(N) / st.pulls)
                scores[a] = st.avg_reward + bonus
        return scores

    def _thompson_scores(self, ctx: ContextStats) -> Dict[str, float]:
        # Placeholder: if you model reward normalized to [0,1], you can use Beta(α,β)
        # Here we just fallback to UCB1 to avoid extra deps.
        return self._ucb1_scores(ctx)

    # ---- public API ----
    def suggest_formats(self, stage: Dict[str, Any], node: Dict[str, Any]) -> List[str]:
        arms = _available_formats(stage, node)
        key = self._ctx_key(stage, node)
        ctx = self._ensure_ctx(arms, key)

        if self.algo == "thompson":
            scores = self._thompson_scores(ctx)
        else:
            scores = self._ucb1_scores(ctx)

        # sort by score (high→low). For +inf first pulls, they'll float to the top.
        ranked = sorted(arms, key=lambda a: scores.get(a, -1e9), reverse=True)
        return ranked

    def choose_format(self, stage: Dict[str, Any], node: Dict[str, Any]) -> str:
        ranked = self.suggest_formats(stage, node)
        return ranked[0] if ranked else "native"

    def record_outcome(
        self,
        stage: Dict[str, Any],
        node: Dict[str, Any],
        chosen_format: str,
        compute_ms: float,
        energy_kj: Optional[float] = None,
        risk: Optional[float] = None,
        reward_mode: str = "neg_latency",
        reward_scale_ms: float = 2000.0,
        reward_alpha: float = 0.7,
    ):
        """
        Update the bandit with an observed outcome.

        reward_mode:
          - "neg_latency": reward = 1 - min(1, compute_ms / reward_scale_ms)  ∈ [0,1]
          - "hybrid": blend latency + energy + risk (weighted)
        reward_alpha: (0..1) used in 'hybrid' to weight latency vs (energy+risk)

        Notes
        -----
        We normalize reward into [0,1] (higher is better). UCB1 doesn't require it,
        but it keeps interpretation consistent and lets you switch to Thompson later.
        """
        arms = _available_formats(stage, node)
        key = self._ctx_key(stage, node)
        ctx = self._ensure_ctx(arms, key)

        if chosen_format not in ctx.arms:
            # unseen arm (e.g., node gained new format)
            ctx.arms[chosen_format] = ArmStats()

        # --- reward shaping ---
        if reward_mode == "hybrid":
            # Latency term
            r_lat = 1.0 - min(1.0, max(0.0, compute_ms / max(1.0, reward_scale_ms)))
            # Energy & risk penalties (map to 0..1 then invert)
            e = energy_kj if energy_kj is not None else 0.0
            r = risk if risk is not None else 0.0
            # crude normalizers
            e_norm = 1.0 - min(1.0, e / 5.0)     # 0 kj → 1, 5+ kj → 0
            r_norm = 1.0 - min(1.0, r)          # risk 0 → 1, risk 1 → 0
            reward = reward_alpha * r_lat + (1.0 - reward_alpha) * (0.5 * e_norm + 0.5 * r_norm)
        else:
            # Pure latency
            reward = 1.0 - min(1.0, max(0.0, compute_ms / max(1.0, reward_scale_ms)))

        # --- update stats ---
        st = ctx.arms[chosen_format]
        st.update(reward=reward, compute_ms=compute_ms, energy_kj=energy_kj, risk=risk)
        ctx.total_pulls += 1
        # write back
        self._state[key]["arms"][chosen_format] = st
        self._state[key]["total_pulls"] = ctx.total_pulls

