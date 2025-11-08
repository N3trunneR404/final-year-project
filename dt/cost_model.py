#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dt/cost_model.py — Latency, transfer, energy, and risk estimators for Fabric DT.

This module is intentionally lightweight and deterministic. It works with the
DTState you already have, and reads the same YAML fields used across your
schemas and generators.

Public API
----------
cm = CostModel(state)

# Per-stage
ms = cm.compute_time_ms(stage, node)
ms = cm.transfer_time_ms(src_node_name, dst_node_name, size_mb)
kj = cm.energy_kj(stage, node, compute_time_ms)
r  = cm.risk_score(stage, node)  # 0..1 (higher = riskier)

# End-to-end (sequential pipeline for MVP; can extend to DAG later)
res = cm.job_cost(job_dict, assignments)  # returns dict with latency_ms, energy_kj, risk, per_stage[]
pen = cm.slo_penalty(deadline_ms, latency_ms)

Conventions
-----------
- Node YAML:
    cpu.cores, cpu.base_ghz, gpu.accel_score, gpu.vram_gb, labels.trust,
    health.{thermal_derate,last_week_crashes}, storage.tbw_pct_used,
    power.tdp_w (optional), dyn.{thermal_derate,down}, caps.{...} (cached by state.py)

- Stage (job YAML):
    id, size_mb, resources.{cpu_cores,mem_gb,gpu_vram_gb}, type,
    allowed_formats, disallowed_formats, hints{io_bound,burstiness}

- Link metrics:
    Taken from state links (effective: speed_gbps, rtt_ms, jitter_ms, loss_pct, ecn),
    or from state's topology defaults if no explicit link entry exists.

Tuning knobs
------------
Adjust the constants in DEFAULTS or pass overrides into CostModel(..., **cfg).
"""

from __future__ import annotations
from typing import Any, Dict, Optional, Tuple, List, Set

from math import isfinite

# No external deps here; DTState is imported only for typing
try:
    from .state import DTState, link_key, safe_float
except Exception:
    # Minimal fallbacks for typing/runtime if imported standalone
    DTState = object  # type: ignore
    def link_key(a: str, b: str) -> str:
        return "|".join(sorted([a, b]))
    def safe_float(x: Any, default: float = 0.0) -> float:
        try:
            return float(x)
        except Exception:
            return default


DEFAULTS = {
    # Compute model
    "MIN_STAGE_MS": 15.0,          # floor latency per stage
    "CPU_UNIT_DIVISOR": 10.0,      # higher → faster (stage base work / (cpu_units/CPU_UNIT_DIVISOR))
    "WASM_PENALTY": 1.35,          # 35% slower than native by default
    "NATIVE_MULT": 1.00,
    "CUDA_BASE_BOOST": 1.0,        # multiplied by (1 + gpu.accel_score/10) then clamped
    "CUDA_MAX_BOOST": 6.0,
    "NPU_TOPS_BOOST_DIV": 10.0,    # 1 + (tops / div) capped
    "NPU_MAX_BOOST": 3.0,

    # Transfer model
    "PROTO_OVERHEAD": 0.85,        # TCP+TLS/etc. overhead factor on throughput
    "LOSS_PENALTY_CEIL": 0.30,     # cap the loss penalty (30% default)
    "DEFAULT_LINK_SPEED_GBPS": 1.0,
    "DEFAULT_RTT_MS": 5.0,
    "DEFAULT_JITTER_MS": 0.5,

    # Energy model
    "DEFAULT_TDP_W": 65.0,         # if node.power.tdp_w missing
    "IDLE_FRACTION": 0.12,         # idle power as fraction of TDP
    "UTIL_TO_POWER_EXP": 0.85,     # sub-linear utilization→power curve

    # Risk model weights
    "R_W_TRUST": 0.25,
    "R_W_SSD_WEAR": 0.15,
    "R_W_CRASH": 0.15,
    "R_W_THERMAL": 0.15,
    "R_W_LINK_LOSS": 0.10,
    "R_W_RELIABILITY": 0.20,

    # SLO penalty
    "SLO_ALPHA": 1.2,              # curvature (>=1 mildly convex)
    "SLO_BETA": 0.002,             # scale (ms^-1)

    # Predictive adjustments
    "RELIABILITY_FAILURE_PEN_MS": 6000.0,
    "RELIABILITY_MIN": 0.35,
    "UTIL_PREDICT_WEIGHT": 0.35,
    "DERATE_LOOKAHEAD_SEC": 600.0,
    "AVAILABILITY_GRACE_SEC": 15.0,
    "LINK_P95_WEIGHT": 0.35,
}


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def merge_stage_details(
    primary: List[Dict[str, Any]] | None,
    cost_entries: List[Dict[str, Any]] | None,
) -> List[Dict[str, Any]]:
    """Combine planner annotations with cost model metrics per stage.

    ``primary`` typically contains rich metadata gathered during planning, such as
    chosen formats, scores, reservation identifiers, and infeasibility reasons.
    ``cost_entries`` is produced by :meth:`CostModel.job_cost` and carries the
    authoritative compute/transfer/energy/risk measurements.

    The helper preserves the ordering from ``cost_entries`` (which mirrors the
    job's stages) while layering any additional keys from ``primary``.  Stages
    present only in ``primary`` are appended afterwards so callers retain their
    annotations even when the cost model skipped them (for instance, when
    planning terminated early).
    """

    primary_list = list(primary or [])
    cost_list = list(cost_entries or [])

    if not primary_list and not cost_list:
        return []

    by_id = {st.get("id"): st for st in primary_list if st.get("id")}
    merged: List[Dict[str, Any]] = []
    seen: Set[Any] = set()

    for entry in cost_list:
        sid = entry.get("id")
        if sid in by_id:
            merged.append({**by_id[sid], **entry})
            seen.add(sid)
        else:
            merged.append(dict(entry))

    for st in primary_list:
        sid = st.get("id")
        if not sid or sid not in seen:
            merged.append(dict(st))

    return merged


class CostModel:
    def __init__(self, state: DTState, **cfg):
        self.state = state
        self.cfg = {**DEFAULTS, **cfg}

    # ---------- helpers ----------

    def _node_cpu_units(self, node: Dict[str, Any]) -> float:
        caps = node.get("caps") or {}
        base = safe_float(caps.get("cpu_units"), 0.0)
        # Apply thermal derate if present (dyn or health)
        dyn = node.get("dyn") or {}
        derate = max(
            safe_float(dyn.get("thermal_derate"), 0.0),
            safe_float((node.get("health") or {}).get("thermal_derate"), 0.0),
        )
        projected = safe_float((node.get("predictive") or {}).get("projected_derate"), None)
        if projected is None and hasattr(self.state, "predict_node_derate"):
            try:
                projected = safe_float(self.state.predict_node_derate(node.get("name", "")), None)
            except Exception:
                projected = None
        if projected is not None:
            derate = max(derate, clamp(projected, 0.0, 1.0))
        return max(0.0, base * (1.0 - clamp(derate, 0.0, 1.0)))

    def _current_reliability(self, node: Dict[str, Any]) -> Optional[float]:
        predictive = node.get("predictive") or {}
        dyn = node.get("dyn") or {}
        reliability = predictive.get("reliability")
        if reliability is None:
            reliability = dyn.get("reliability")
        if reliability is None and hasattr(self.state, "node_reliability"):
            try:
                reliability = self.state.node_reliability(node.get("name", ""))
            except Exception:
                reliability = None
        if reliability is None:
            return None
        return clamp(safe_float(reliability, 0.95), 0.0, 1.0)

    def _current_availability(self, node: Dict[str, Any]) -> Optional[float]:
        predictive = node.get("predictive") or {}
        dyn = node.get("dyn") or {}
        availability = predictive.get("availability_window_sec")
        if availability is None:
            availability = dyn.get("availability_window_sec")
        if availability is None and hasattr(self.state, "node_availability_window"):
            try:
                availability = self.state.node_availability_window(node.get("name", ""))
            except Exception:
                availability = None
        if availability is None:
            return None
        return max(0.0, safe_float(availability, 0.0))

    def _accel_multiplier(self, node: Dict[str, Any], stage: Dict[str, Any]) -> float:
        """
        Boost compute if the node supports a preferred format requested by the stage.
        """
        fmts = set(node.get("formats_supported") or [])
        allowed = set(stage.get("allowed_formats") or [])
        disallowed = set(stage.get("disallowed_formats") or [])
        # If allowed set exists and we fail it entirely, return a huge penalty
        if allowed and not (fmts & allowed):
            return 0.5  # still allow (planner can decide infeasible elsewhere)

        mult = 1.0

        # CUDA
        if ("cuda" in fmts) and ("cuda" not in disallowed) and (not allowed or "cuda" in allowed):
            score = safe_float((node.get("gpu") or {}).get("accel_score"), 0.0)
            cuda = self.cfg["CUDA_BASE_BOOST"] * (1.0 + score / 10.0)
            mult = max(mult, clamp(cuda, 1.0, self.cfg["CUDA_MAX_BOOST"]))

        # NPU
        if ("npu" in fmts) and ("npu" not in disallowed) and (not allowed or "npu" in allowed):
            tops = safe_float((node.get("accelerators") or {}).get("npu_tops"), 0.0)
            npu = 1.0 + (tops / self.cfg["NPU_TOPS_BOOST_DIV"])
            mult = max(mult, clamp(npu, 1.0, self.cfg["NPU_MAX_BOOST"]))

        # WASM penalty (if stage prefers wasm or node only offers wasm/native)
        if "wasm" in fmts and (allowed == {"wasm"} or ("wasm" in allowed and "native" not in allowed)):
            mult = mult / self.cfg["WASM_PENALTY"]

        # Native OK
        return mult

    def _stage_base_work(self, stage: Dict[str, Any]) -> float:
        """
        A rough scalar for 'how big' a stage is. You can later switch to FLOP/byte hints.
        """
        size_mb = safe_float(stage.get("size_mb"), 10.0)
        cpu_req = safe_float((stage.get("resources") or {}).get("cpu_cores"), 1.0)
        # scale: data-size contributes, cpu cores requested pushes it up more
        base = size_mb * 2.0 + cpu_req * 120.0
        if (stage.get("hints") or {}).get("io_bound", False):
            base *= 0.85  # IO-bound likely less CPU compute (but transfer dominates)
        return max(self.cfg["MIN_STAGE_MS"], base)

    # ---------- compute / transfer ----------

    def compute_time_ms(self, stage: Dict[str, Any], node: Dict[str, Any]) -> float:
        if (node.get("dyn") or {}).get("down", False):
            return float("inf")
        cpu_units = self._node_cpu_units(node)
        if cpu_units <= 1e-9:
            return float("inf")
        work = self._stage_base_work(stage)
        accel = self._accel_multiplier(node, stage)
        # Smaller score => faster; divide by cpu scale and accel
        t = work / max(1.0, (cpu_units / self.cfg["CPU_UNIT_DIVISOR"])) / max(1.0, accel)
        # Respect minimal latency floor
        predictive = node.get("predictive") or {}
        dyn = node.get("dyn") or {}
        util_forecast = predictive.get("util_forecast")
        if util_forecast is None:
            util_forecast = dyn.get("util_forecast")
        try:
            if util_forecast is None and hasattr(self.state, "predictive_overview"):
                overview = self.state.predictive_overview()
                util_forecast = (
                    (overview.get("nodes", {}).get(node.get("name", ""), {}) or {}).get("util_forecast")
                )
        except Exception:
            util_forecast = None
        if util_forecast is not None:
            t *= 1.0 + clamp(safe_float(util_forecast, 0.0), 0.0, 1.0) * self.cfg["UTIL_PREDICT_WEIGHT"]

        reliability = self._current_reliability(node)
        if reliability is not None:
            if reliability < self.cfg["RELIABILITY_MIN"]:
                return float("inf")
            t += (1.0 - reliability) * self.cfg["RELIABILITY_FAILURE_PEN_MS"]

        availability = self._current_availability(node)
        if availability is not None:
            available_ms = max(0.0, availability - self.cfg["AVAILABILITY_GRACE_SEC"]) * 1000.0
            if available_ms > 0 and t > available_ms:
                return float("inf")

        return max(self.cfg["MIN_STAGE_MS"], t)

    def _effective_link_metrics(self, a: str, b: str) -> Dict[str, float]:
        k = link_key(a, b)
        L = self.state.links_by_key.get(k)
        if L:
            eff = self.state._effective_link(L)  # uses defaults if needed
            return {
                "speed_gbps": eff.get("speed_gbps", self.cfg["DEFAULT_LINK_SPEED_GBPS"]),
                "rtt_ms": eff.get("rtt_ms", self.cfg["DEFAULT_RTT_MS"]),
                "jitter_ms": eff.get("jitter_ms", self.cfg["DEFAULT_JITTER_MS"]),
                "loss_pct": eff.get("loss_pct", 0.0),
                "down": bool(eff.get("down", False)),
            }
        # Fall back to topology defaults if no explicit link
        netdef = (self.state.defaults.get("network") or {})
        return {
            "speed_gbps": safe_float(netdef.get("speed_gbps"), self.cfg["DEFAULT_LINK_SPEED_GBPS"]),
            "rtt_ms": safe_float(netdef.get("rtt_ms"), self.cfg["DEFAULT_RTT_MS"]),
            "jitter_ms": safe_float(netdef.get("jitter_ms"), self.cfg["DEFAULT_JITTER_MS"]),
            "loss_pct": safe_float(netdef.get("loss_pct"), 0.0),
            "down": False,
        }

    def transfer_time_ms(self, src: str, dst: str, size_mb: float) -> float:
        if size_mb <= 0 or src == dst:
            return 0.0
        m = self._effective_link_metrics(src, dst)
        if m["down"]:
            return float("inf")
        mbps_phy = m["speed_gbps"] * 1000.0
        loss_pen = 1.0 - clamp(m["loss_pct"] / 100.0, 0.0, self.cfg["LOSS_PENALTY_CEIL"])
        eff_mbps = mbps_phy * self.cfg["PROTO_OVERHEAD"] * loss_pen
        xfer = (size_mb * 8.0) / max(1.0, eff_mbps) * 1000.0  # ms
        base_latency = xfer + m["rtt_ms"] + m["jitter_ms"]
        variability = None
        if hasattr(self.state, "link_variability"):
            try:
                variability = self.state.link_variability(src, dst)
            except Exception:
                variability = None
        if variability:
            p95 = variability.get("latency_p95_ms")
            if p95 is not None:
                base_latency = max(base_latency, safe_float(p95, base_latency))
            else:
                base_latency += safe_float(variability.get("jitter_ms"), 0.0) * self.cfg["LINK_P95_WEIGHT"]
            if safe_float(variability.get("loss_pct"), 0.0) >= 30.0:
                return float("inf")
        return base_latency

    # ---------- energy & risk ----------

    def energy_kj(self, stage: Dict[str, Any], node: Dict[str, Any], compute_time_ms: float) -> float:
        """Very rough: (idle+active) power × time."""
        power = (node.get("power") or {})
        tdp = safe_float(power.get("tdp_w"), self.cfg["DEFAULT_TDP_W"])
        # Util proxy: requested cores / max cores (bounded) and size scaling
        req = safe_float((stage.get("resources") or {}).get("cpu_cores"), 1.0)
        max_cores = safe_float((node.get("caps") or {}).get("max_cpu_cores"), 1.0)
        util = clamp(req / max(1.0, max_cores), 0.05, 1.0)

        # Thermal derate increases power waste a bit
        der = max(
            safe_float((node.get("dyn") or {}).get("thermal_derate"), 0.0),
            safe_float((node.get("health") or {}).get("thermal_derate"), 0.0),
        )
        predictive = node.get("predictive") or {}
        util_forecast = predictive.get("util_forecast")
        if util_forecast is None:
            util_forecast = (node.get("dyn") or {}).get("util_forecast")
        util_eff = clamp(util * (1.0 + 0.2 * der), 0.0, 1.0)
        if util_forecast is not None:
            util_eff = clamp(util_eff * (1.0 + clamp(safe_float(util_forecast, 0.0), 0.0, 1.0) * 0.25), 0.0, 1.0)

        idle_w = tdp * self.cfg["IDLE_FRACTION"]
        active_w = (tdp - idle_w) * (util_eff ** self.cfg["UTIL_TO_POWER_EXP"])
        watts = idle_w + active_w
        sec = compute_time_ms / 1000.0
        kj = watts * sec / 1000.0
        return max(0.0, kj)

    def risk_score(self, stage: Dict[str, Any], node: Dict[str, Any], link_loss_pct: float = 0.0) -> float:
        """Blend node trust inverse, SSD wear, crashiness, thermal issues, and link loss."""
        labels = node.get("labels") or {}
        trust = None
        try:
            trust = float(labels.get("trust"))
        except Exception:
            trust = None
        trust_term = 1.0 - clamp(trust if trust is not None else 0.8, 0.0, 1.0)

        ssd_wear = safe_float((node.get("storage") or {}).get("tbw_pct_used"), 0.0) / 100.0
        crashes = safe_float((node.get("health") or {}).get("last_week_crashes"), 0.0)
        crash_term = clamp(crashes / 5.0, 0.0, 1.0)  # 5+ crashes → max

        dyn = node.get("dyn") or {}
        thermal = max(
            safe_float(dyn.get("thermal_derate"), 0.0),
            safe_float((node.get("health") or {}).get("thermal_derate"), 0.0),
        )

        reliability = self._current_reliability(node)
        reliability_term = 1.0 - reliability if reliability is not None else 0.0

        availability = self._current_availability(node)
        availability_term = 0.0
        if availability is not None:
            # < 2 minutes availability increases risk significantly
            availability_term = clamp(max(0.0, 120.0 - availability) / 120.0, 0.0, 1.0)

        link_term = clamp(link_loss_pct / 5.0, 0.0, 1.0)  # 5%+ loss → max

        w = self.cfg
        r = (
            w["R_W_TRUST"]   * trust_term +
            w["R_W_SSD_WEAR"]* clamp(ssd_wear, 0.0, 1.0) +
            w["R_W_CRASH"]   * crash_term +
            w["R_W_THERMAL"] * clamp(thermal, 0.0, 1.0) +
            w["R_W_LINK_LOSS"] * link_term +
            w["R_W_RELIABILITY"] * reliability_term +
            0.15 * availability_term
        )
        return clamp(r, 0.0, 1.0)

    # ---------- end-to-end job ----------

    def job_cost(self, job: Dict[str, Any], assignments: Dict[str, str]) -> Dict[str, Any]:
        """
        Compute sequential pipeline cost for a single job using given stage->node assignments.
        Returns:
        {
          "latency_ms": float,
          "energy_kj": float,
          "risk": float,                 # aggregated (mean) risk
          "per_stage": [
              {"id": "...", "node": "...", "compute_ms": ..., "xfer_ms": ..., "energy_kj": ..., "risk": ...}
          ]
        }
        """
        stages = job.get("stages") or []
        total_ms = 0.0
        total_kj = 0.0
        risks: List[float] = []
        results: List[Dict[str, Any]] = []

        prev_node = None
        for st in stages:
            sid = st.get("id")
            node_name = assignments.get(sid)
            if node_name is None:
                # infeasible
                results.append({"id": sid, "node": None, "compute_ms": float("inf"), "xfer_ms": 0.0, "energy_kj": 0.0, "risk": 1.0})
                total_ms = float("inf")
                continue
            node = self.state.get_node(node_name)
            if not node:
                results.append({"id": sid, "node": node_name, "compute_ms": float("inf"), "xfer_ms": 0.0, "energy_kj": 0.0, "risk": 1.0})
                total_ms = float("inf")
                continue

            # Transfer from previous stage output (use st.size_mb as proxy)
            size_mb = safe_float(st.get("size_mb"), 10.0)
            if prev_node is None:
                xfer_ms = 0.0
                link_loss = 0.0
            else:
                m = self._effective_link_metrics(prev_node, node_name)
                xfer_ms = self.transfer_time_ms(prev_node, node_name, size_mb)
                link_loss = m.get("loss_pct", 0.0)

            comp_ms = self.compute_time_ms(st, node)
            en_kj = self.energy_kj(st, node, comp_ms)
            risk = self.risk_score(st, node, link_loss_pct=link_loss)
            reliability = self._current_reliability(node)
            availability = self._current_availability(node)

            results.append({
                "id": sid,
                "node": node_name,
                "compute_ms": round(comp_ms, 3),
                "xfer_ms": round(xfer_ms, 3),
                "energy_kj": round(en_kj, 5),
                "risk": round(risk, 4),
                "reliability": None if reliability is None else round(reliability, 4),
                "availability_window_sec": None if availability is None else round(availability, 3),
            })

            if isfinite(comp_ms) and isfinite(xfer_ms):
                total_ms += comp_ms + xfer_ms
                total_kj += en_kj
                risks.append(risk)
            else:
                total_ms = float("inf")

            prev_node = node_name

        agg_risk = sum(risks) / max(1, len(risks)) if risks else 1.0
        reliability_vals = [
            entry.get("reliability")
            for entry in results
            if entry.get("reliability") is not None
        ]
        avg_reliability = (
            sum(reliability_vals) / len(reliability_vals)
            if reliability_vals
            else None
        )

        return {
            "latency_ms": round(total_ms, 3),
            "energy_kj": round(total_kj, 5),
            "risk": round(agg_risk, 4),
            "avg_reliability": None if avg_reliability is None else round(avg_reliability, 4),
            "per_stage": results,
        }

    # ---------- SLO penalty ----------

    def slo_penalty(self, deadline_ms: float, latency_ms: float) -> float:
        """
        Smooth, monotonic penalty ≥0. 0 if latency <= deadline; grows smoothly after.
        penalty = ((lat/ddl) ** alpha - 1)+ clamped at 0, then scaled.
        """
        if deadline_ms <= 0 or not isfinite(latency_ms):
            return 0.0
        ratio = clamp(latency_ms / max(1.0, deadline_ms), 0.0, 100.0)
        if ratio <= 1.0:
            return 0.0
        a = self.cfg["SLO_ALPHA"]
        b = self.cfg["SLO_BETA"]
        return (ratio ** a - 1.0) / max(1e-6, b)

