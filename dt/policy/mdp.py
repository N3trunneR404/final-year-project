"""Markov decision process planner with reinforcement learning assistance.

The :class:`MarkovPlanner` treats a sequential Fabric job as a finite-horizon
MDP. Each stage constitutes a decision step where the agent chooses a node that
meets the resource/format constraints. Immediate costs combine latency, energy,
risk, utilisation, and network penalties using a weighted Tchebycheff
aggregation so the optimiser respects the Pareto shape rather than collapsing
metrics into an ad-hoc scalar.  The dynamic program backs up an expected value
function that discounts future stages and folds in failure penalties derived
from per-stage risk.

To adapt over time, the planner can be paired with the lightweight
:class:`dt.policy.rl_stub.RLPolicy`.  Learned Q-values become an additional
objective (maximised) and are updated with the predicted latency/energy/risk for
any plan that is produced.  When a plan is committed (``dry_run=False``) the
Q-table is persisted so future runs benefit from the accumulated experience.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Tuple

from dt.cost_model import CostModel, merge_stage_details
from dt.policy.rl_stub import RLPolicy, _stage_sig as rl_stage_sig
from dt.state import DTState, safe_float


@dataclass
class PlannerWeights:
    latency: float = 0.38
    energy: float = 0.16
    risk: float = 0.18
    load: float = 0.12
    network: float = 0.10
    federation: float = 0.06
    rl: float = 0.06


class MarkovPlanner:
    """Solve a sequential placement problem via dynamic programming."""

    def __init__(
        self,
        state: DTState,
        cost_model: CostModel,
        rl_policy: Optional[RLPolicy] = None,
        *,
        gamma: float = 0.9,
        failure_penalty: float = 8.0,
        redundancy: int = 2,
        weights: Optional[PlannerWeights] = None,
    ) -> None:
        self.state = state
        self.cm = cost_model
        self.gamma = float(gamma)
        self.failure_penalty = float(failure_penalty)
        self.redundancy = max(1, int(redundancy))
        self.weights = weights or PlannerWeights()
        self.rl_policy = rl_policy

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plan_job(self, job: Dict[str, Any], dry_run: bool = False) -> Dict[str, Any]:
        self._fed_snapshot: Dict[str, Dict[str, Any]] = {}
        stages = list(job.get("stages") or [])
        if not stages:
            return {
                "job_id": job.get("id"),
                "assignments": {},
                "per_stage": [],
                "reservations": [],
                "shadow_assignments": {},
                "latency_ms": 0.0,
                "energy_kj": 0.0,
                "risk": 0.0,
                "strategy": "rl-markov",
                "dry_run": dry_run,
                "infeasible": True,
                "reason": "no_stages",
                "ts": int(time.time() * 1000),
            }

        nodes = self.state.nodes_for_planner()
        fed_overview = self.state.federations_overview()
        node_to_fed = fed_overview.get("node_federations") or {}
        fed_stats = {
            entry.get("name"): dict(entry)
            for entry in (fed_overview.get("federations") or [])
        }
        self._fed_snapshot = {k: dict(v) for k, v in fed_stats.items()}

        solved = self._solve_job(tuple(range(len(stages))), nodes, tuple(stages))
        assignments = dict(solved.assignments)
        stage_details = solved.stage_details
        shadow_assignments: Dict[str, List[str]] = {}
        reservations: List[Dict[str, str]] = []
        infeasible = False

        prev_node = None
        for entry in stage_details:
            sid = entry.get("id")
            node_name = entry.get("node")
            if not node_name:
                infeasible = True
                prev_node = None
                continue

            # Reserve capacity if required
            assigned = True
            if not dry_run:
                res = entry.get("requested_resources") or {}
                req = {"node": node_name, **res}
                res_id = self.state.reserve(req)
                if res_id:
                    reservations.append({"node": node_name, "reservation_id": res_id})
                    entry["reservation_id"] = res_id
                else:
                    entry["infeasible"] = True
                    entry.setdefault("reason", "reservation_failed")
                    entry["node"] = None
                    entry["prev_node"] = None
                    infeasible = True
                    if sid:
                        assignments.pop(sid, None)
                    assigned = False

            fallbacks = entry.get("fallbacks") or []
            if sid:
                shadow_assignments[sid] = fallbacks
            prev_node = node_name if assigned else None

        # Merge with authoritative cost model figures
        cost = self.cm.job_cost(job, assignments)
        merged = merge_stage_details(stage_details, cost.get("per_stage"))

        ddl = safe_float(job.get("deadline_ms"), 0.0)
        slo_penalty = self.cm.slo_penalty(
            ddl, cost.get("latency_ms", float("inf"))
        ) if ddl > 0 else 0.0

        # Federation spread metrics
        unique_feds = {
            node_to_fed.get(node) or self.state.federation_for_node(node) or "global"
            for node in assignments.values()
        }
        spread = len(unique_feds) / max(1, len(stages))
        fallback_ratio = sum(1 for v in shadow_assignments.values() if v) / max(1, len(stages))
        crossfed = 0
        for entry in stage_details:
            fed = entry.get("federation")
            for fb, fb_fed in zip(entry.get("fallbacks") or [], entry.get("fallback_federations") or []):
                if fb and fb_fed and fb_fed != fed:
                    crossfed += 1
        crossfed_ratio = crossfed / max(1, len(stages))

        result = {
            "job_id": job.get("id"),
            "assignments": assignments,
            "reservations": reservations,
            "shadow_assignments": shadow_assignments,
            "per_stage": merged,
            "latency_ms": cost.get("latency_ms"),
            "energy_kj": cost.get("energy_kj"),
            "risk": cost.get("risk"),
            "deadline_ms": ddl or None,
            "slo_penalty": slo_penalty,
            "infeasible": infeasible or (cost.get("latency_ms") == float("inf")),
            "strategy": "rl-markov",
            "dry_run": dry_run,
            "federation_spread": round(spread, 4),
            "federations_in_use": sorted(unique_feds),
            "resilience_score": round(fallback_ratio, 4),
            "cross_federation_fallback_ratio": round(crossfed_ratio, 4),
            "projected_federations": list(fed_stats.values()),
            "ts": int(time.time() * 1000),
            "mdp_value": round(solved.cost, 6),
        }

        if self.rl_policy:
            self._update_rl(job, stage_details)
            if not dry_run:
                self.rl_policy.save()

        self._fed_snapshot = {}

        return result

    # ------------------------------------------------------------------
    # Internal structures
    # ------------------------------------------------------------------

    @dataclass(frozen=True)
    class _Solution:
        assignments: Dict[str, str]
        stage_details: List[Dict[str, Any]]
        cost: float

    def _solve_job(
        self,
        stage_indices: Tuple[int, ...],
        nodes: Dict[str, Dict[str, Any]],
        stages: Tuple[Dict[str, Any], ...],
    ) -> "MarkovPlanner._Solution":
        cache: Dict[Tuple[int, Optional[str]], MarkovPlanner._Solution] = {}

        @lru_cache(maxsize=None)
        def recurse(idx: int, prev_node: Optional[str]) -> MarkovPlanner._Solution:
            if idx >= len(stage_indices):
                return MarkovPlanner._Solution(assignments={}, stage_details=[], cost=0.0)

            stage = stages[stage_indices[idx]]
            sid = stage.get("id")
            candidates = self._enumerate_candidates(stage, prev_node, nodes)
            if not candidates:
                detail = {
                    "id": sid,
                    "node": None,
                    "infeasible": True,
                    "reason": "no_feasible_node",
                    "fallbacks": [],
                    "fallback_federations": [],
                }
                return MarkovPlanner._Solution(
                    assignments={}, stage_details=[detail], cost=self.failure_penalty
                )

            ideals, nadirs = self._metric_bounds(candidates.values())
            ranked: List[Tuple[float, Dict[str, Any]]] = []

            for name, metrics in candidates.items():
                child = recurse(idx + 1, name)
                immediate, breakdown = self._score_candidate(metrics, ideals, nadirs)
                future_cost = child.cost
                expected_tail = self.gamma * (
                    (1.0 - metrics["risk"]) * future_cost + metrics["risk"] * self.failure_penalty
                )
                total = immediate + expected_tail
                ranked.append(
                    (
                        total,
                        {
                            **metrics,
                            "child": child,
                            "immediate": immediate,
                            "breakdown": breakdown,
                            "expected_future": expected_tail,
                        },
                    )
                )

            ranked.sort(key=lambda item: item[0])
            best_total, best_payload = ranked[0]
            metrics = best_payload
            child = metrics["child"]
            best_name = metrics["node"]

            assignments = dict(child.assignments)
            if sid:
                assignments[sid] = best_name

            fallback_nodes: List[str] = []
            fallback_feds: List[str] = []
            for _, payload in ranked[1 : 1 + self.redundancy - 1]:
                fallback_nodes.append(payload["node"])
                fallback_feds.append(payload.get("federation"))

            detail = {
                "id": sid,
                "node": best_name,
                "federation": metrics.get("federation"),
                "fallbacks": fallback_nodes,
                "fallback_federations": fallback_feds,
                "rl_value": metrics.get("rl_value"),
                "expected_cost": round(best_total, 6),
                "immediate_cost": round(metrics["immediate"], 6),
                "score_breakdown": {
                    k: round(v, 6) for k, v in (metrics["breakdown"] or {}).items()
                },
                "candidate_details": [
                    self._serialise_candidate(total, payload)
                    for total, payload in ranked
                ],
                "requested_resources": metrics.get("requested_resources", {}),
                "compute_ms": round(metrics.get("compute_ms", 0.0), 6),
                "xfer_ms": round(metrics.get("xfer_ms", 0.0), 6),
                "energy_kj": round(metrics.get("energy", 0.0), 6),
                "risk": round(metrics.get("risk", 0.0), 6),
                "load_factor": round(metrics.get("load", 0.0), 6),
                "network_penalty": round(metrics.get("network", 0.0), 6),
                "federation_penalty": round(metrics.get("federation_penalty", 0.0), 6),
                "format": metrics.get("format"),
                "prev_node": prev_node,
                "link_loss_pct": round(metrics.get("link_loss_pct", 0.0), 6),
            }

            return MarkovPlanner._Solution(
                assignments=assignments,
                stage_details=[detail] + child.stage_details,
                cost=best_total,
            )

        final = recurse(0, None)
        cache.update({})  # appease linters; cache kept for clarity
        return final

    # ------------------------------------------------------------------
    # Candidate analysis helpers
    # ------------------------------------------------------------------

    def _enumerate_candidates(
        self,
        stage: Dict[str, Any],
        prev_node: Optional[str],
        nodes: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        size_mb = safe_float(stage.get("size_mb"), 10.0)
        for name, node in nodes.items():
            if not self._fits(node, stage):
                continue
            compute_ms = self.cm.compute_time_ms(stage, node)
            if math.isinf(compute_ms):
                continue
            if prev_node:
                link = self.cm._effective_link_metrics(prev_node, name)  # type: ignore[attr-defined]
                xfer_ms = self.cm.transfer_time_ms(prev_node, name, size_mb)
                link_loss = safe_float(link.get("loss_pct"), 0.0)
            else:
                xfer_ms = 0.0
                link_loss = 0.0
            latency = compute_ms + xfer_ms
            energy = self.cm.energy_kj(stage, node, compute_ms)
            risk = self.cm.risk_score(stage, node, link_loss)
            load = self._load_factor(node, stage)
            network_pen = self._network_penalty(xfer_ms, link_loss)
            fed_name = self.state.federation_for_node(name) or "global"
            fed_pen = self._federation_penalty(fed_name)
            rl_val = self.rl_policy.value(self._stage_sig(stage), name) if self.rl_policy else 0.0
            format_choice = self._choose_format(stage, node)
            out[name] = {
                "node": name,
                "compute_ms": compute_ms,
                "xfer_ms": xfer_ms,
                "latency": latency,
                "energy": energy,
                "risk": clamp01(risk),
                "load": clamp01(load),
                "network": clamp01(network_pen),
                "federation_penalty": clamp01(fed_pen),
                "federation": fed_name,
                "rl_penalty": clamp01(self._rl_penalty(rl_val)),
                "rl_value": rl_val,
                "format": format_choice,
                "requested_resources": self._resource_request(stage),
                "link_loss_pct": link_loss,
            }
        return out

    def _metric_bounds(self, candidates: Iterable[Dict[str, Any]]) -> Tuple[Dict[str, float], Dict[str, float]]:
        metrics = [c for c in candidates]
        keys = ["latency", "energy", "risk", "load", "network", "federation_penalty", "rl_penalty"]
        ideals = {k: float("inf") for k in keys}
        nadirs = {k: float("-inf") for k in keys}
        for cand in metrics:
            for k in keys:
                val = float(cand.get(k, 0.0))
                ideals[k] = min(ideals[k], val)
                nadirs[k] = max(nadirs[k], val)
        for k in keys:
            if ideals[k] == float("inf"):
                ideals[k] = 0.0
            if nadirs[k] == float("-inf"):
                nadirs[k] = ideals[k]
        return ideals, nadirs

    def _score_candidate(
        self,
        metrics: Dict[str, Any],
        ideals: Dict[str, float],
        nadirs: Dict[str, float],
    ) -> Tuple[float, Dict[str, float]]:
        weights = self.weights
        rho = 0.12
        breakdown: Dict[str, float] = {}
        components: List[float] = []
        for key, weight in [
            ("latency", weights.latency),
            ("energy", weights.energy),
            ("risk", weights.risk),
            ("load", weights.load),
            ("network", weights.network),
            ("federation_penalty", weights.federation),
            ("rl_penalty", weights.rl),
        ]:
            span = max(1e-9, nadirs[key] - ideals[key])
            norm = (float(metrics.get(key, 0.0)) - ideals[key]) / span
            norm = max(0.0, norm)
            comp = weight * norm
            breakdown[key] = comp
            components.append(comp)
        aggregate = max(components) + rho * sum(c ** 2 for c in components)
        return aggregate, breakdown

    # ------------------------------------------------------------------
    # Feature helpers
    # ------------------------------------------------------------------

    def _resource_request(self, stage: Dict[str, Any]) -> Dict[str, float]:
        res = stage.get("resources") or {}
        return {
            "cpu_cores": safe_float(res.get("cpu_cores"), 0.0),
            "mem_gb": safe_float(res.get("mem_gb"), 0.0),
            "gpu_vram_gb": safe_float(res.get("gpu_vram_gb"), 0.0),
        }

    def _fits(self, node: Dict[str, Any], stage: Dict[str, Any]) -> bool:
        if (node.get("dyn") or {}).get("down", False):
            return False
        eff = node.get("effective") or {}
        need = self._resource_request(stage)
        if eff.get("free_cpu_cores", 0.0) + 1e-9 < need["cpu_cores"]:
            return False
        if eff.get("free_mem_gb", 0.0) + 1e-9 < need["mem_gb"]:
            return False
        if eff.get("free_gpu_vram_gb", 0.0) + 1e-9 < need["gpu_vram_gb"]:
            return False
        return self._supports_formats(node, stage)

    def _supports_formats(self, node: Dict[str, Any], stage: Dict[str, Any]) -> bool:
        allowed = stage.get("allowed_formats") or []
        disallowed = stage.get("disallowed_formats") or []
        fmts = node.get("formats_supported") or []
        if any(fmt in fmts for fmt in disallowed):
            return False
        if not allowed:
            return True
        return any(fmt in fmts for fmt in allowed)

    def _choose_format(self, stage: Dict[str, Any], node: Dict[str, Any]) -> Optional[str]:
        allowed = stage.get("allowed_formats") or []
        if not allowed:
            return None
        fmts = node.get("formats_supported") or []
        for fmt in allowed:
            if fmt in fmts:
                return fmt
        return allowed[0] if allowed else None

    def _load_factor(self, node: Dict[str, Any], stage: Dict[str, Any]) -> float:
        eff = node.get("effective") or {}
        loads: List[float] = []
        need = self._resource_request(stage)
        total_cpu = safe_float(eff.get("max_cpu_cores"), 0.0)
        if total_cpu > 0:
            free_cpu = max(0.0, safe_float(eff.get("free_cpu_cores"), 0.0) - need["cpu_cores"])
            loads.append((total_cpu - free_cpu) / max(total_cpu, 1e-6))
        total_mem = safe_float(eff.get("max_mem_gb"), 0.0)
        if total_mem > 0:
            free_mem = max(0.0, safe_float(eff.get("free_mem_gb"), 0.0) - need["mem_gb"])
            loads.append((total_mem - free_mem) / max(total_mem, 1e-6))
        total_vram = safe_float(eff.get("max_gpu_vram_gb"), 0.0)
        if total_vram > 0:
            free_vram = max(0.0, safe_float(eff.get("free_gpu_vram_gb"), 0.0) - need["gpu_vram_gb"])
            loads.append((total_vram - free_vram) / max(total_vram, 1e-6))
        if not loads:
            return 0.0
        return sum(clamp01(x) for x in loads) / len(loads)

    def _network_penalty(self, xfer_ms: float, loss_pct: float) -> float:
        # Blend transfer latency and loss into 0..1 window
        xfer_term = min(1.0, xfer_ms / 500.0)
        loss_term = min(1.0, loss_pct / 5.0)
        return (0.7 * xfer_term) + (0.3 * loss_term)

    def _federation_penalty(self, federation: str) -> float:
        stats = getattr(self, "_fed_snapshot", None) or {}
        entry = stats.get(federation)
        if entry is not None:
            return safe_float(entry.get("load_factor"), 0.0)
        overview = self.state.federations_overview()
        for item in overview.get("federations") or []:
            if item.get("name") == federation:
                return safe_float(item.get("load_factor"), 0.0)
        return 0.0

    def _rl_penalty(self, value: float) -> float:
        # Higher Q-values should decrease cost, so convert to a small penalty
        if value <= 0:
            return 0.5
        # map through a sigmoid-ish curve to (0, 0.5)
        return max(0.0, 0.5 - (math.tanh(value / 10.0) * 0.5))

    def _stage_sig(self, stage: Dict[str, Any]) -> str:
        # Reuse RLPolicy bucketing so lookups align
        return rl_stage_sig(stage)

    def _serialise_candidate(self, total: float, payload: Dict[str, Any]) -> Dict[str, Any]:
        metrics = payload.get("metrics", {})
        return {
            "node": payload.get("node"),
            "total_cost": round(float(total), 6),
            "immediate_cost": round(float(payload.get("immediate", 0.0)), 6),
            "expected_future": round(float(payload.get("expected_future", 0.0)), 6),
            "latency_ms": round(float(metrics.get("latency", 0.0)), 6),
            "compute_ms": round(float(metrics.get("compute_ms", 0.0)), 6),
            "xfer_ms": round(float(metrics.get("xfer_ms", 0.0)), 6),
            "energy_kj": round(float(metrics.get("energy", 0.0)), 6),
            "risk": round(float(metrics.get("risk", 0.0)), 6),
            "load_factor": round(float(metrics.get("load", 0.0)), 6),
            "network_penalty": round(float(metrics.get("network", 0.0)), 6),
            "federation_penalty": round(float(metrics.get("federation_penalty", 0.0)), 6),
            "rl_penalty": round(float(metrics.get("rl_penalty", 0.0)), 6),
            "rl_value": round(float(metrics.get("rl_value", 0.0)), 6),
            "federation": metrics.get("federation"),
            "score_breakdown": {
                k: round(float(v), 6) for k, v in (payload.get("breakdown") or {}).items()
            },
        }

    # ------------------------------------------------------------------
    # Reinforcement updates
    # ------------------------------------------------------------------

    def _update_rl(self, job: Dict[str, Any], stage_details: List[Dict[str, Any]]):
        if not self.rl_policy:
            return
        stages = list(job.get("stages") or [])
        if not stages:
            return
        details_by_id = {entry.get("id"): entry for entry in stage_details if entry.get("id")}
        ordered_details: List[Dict[str, Any]] = []
        for stage in stages:
            sid = stage.get("id")
            if sid and sid in details_by_id:
                ordered_details.append(details_by_id[sid])

        prev_node = None
        for idx, stage in enumerate(stages):
            detail = ordered_details[idx] if idx < len(ordered_details) else None
            if not detail:
                continue
            node = detail.get("node")
            if not node:
                prev_node = None
                continue
            compute_ms = safe_float(detail.get("compute_ms"), 0.0)
            energy = safe_float(detail.get("energy_kj"), 0.0)
            risk = safe_float(detail.get("risk"), 0.0)

            next_stage = stages[idx + 1] if idx + 1 < len(stages) else None
            next_detail = ordered_details[idx + 1] if idx + 1 < len(ordered_details) else None
            if next_stage and next_detail:
                next_candidates = {
                    cand.get("node"): {}
                    for cand in (next_detail.get("candidate_details") or [])
                    if cand.get("node")
                }
                next_node = next_detail.get("node")
            else:
                next_candidates = None
                next_node = None

            self.rl_policy.record_transition(
                stage,
                prev_node,
                node,
                compute_ms,
                energy_kj=energy,
                risk=risk,
                next_stage=next_stage,
                next_candidates=next_candidates,
                next_node=next_node,
            )
            prev_node = node


# ----------------------------------------------------------------------
# Utility
# ----------------------------------------------------------------------


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))
