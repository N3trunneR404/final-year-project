#!/usr/bin/env python3
"""Background controllers that keep the twin pseudo-autonomous."""

from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .state import DTState, safe_float


@dataclass
class ShadowReservation:
    node: str
    reservation_id: str
    created_ts: float = field(default_factory=lambda: time.time())


@dataclass
class ManagedStage:
    job_id: str
    stage_id: str
    resources: Dict[str, float]
    primary_node: str
    reservation_id: str
    fallbacks: List[Dict[str, Any]] = field(default_factory=list)
    shadow_reservations: List[ShadowReservation] = field(default_factory=list)
    redundancy: int = 1
    degraded_ticks: int = 0
    last_transition: float = field(default_factory=lambda: time.time())


class SelfHealingController:
    """Promotes fallback placements when primaries degrade or fail."""

    def __init__(
        self,
        state: DTState,
        *,
        poll_interval: float = 2.0,
        reliability_threshold: float = 0.75,
        availability_threshold_sec: float = 90.0,
        stability_window: int = 2,
    ) -> None:
        self.state = state
        self.poll_interval = max(0.5, poll_interval)
        self.reliability_threshold = reliability_threshold
        self.availability_threshold_sec = availability_threshold_sec
        self.stability_window = max(1, stability_window)

        self._lock = threading.RLock()
        self._managed: Dict[str, Dict[str, ManagedStage]] = {}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="SelfHealing", daemon=True)
        self._thread.start()

    # ------------------------- public API -------------------------

    def register_plan(self, job: Dict[str, Any], plan: Dict[str, Any]) -> None:
        stages = job.get("stages") or []
        stage_by_id = {st.get("id"): st for st in stages if st.get("id")}

        job_id = plan.get("job_id") or job.get("id") or f"job-{int(time.time()*1000)}"
        redundancy = max(1, int(job.get("redundancy") or plan.get("redundancy") or 1))

        with self._lock:
            bucket = self._managed.setdefault(job_id, {})
            for stage_entry in plan.get("per_stage", []):
                stage_id = stage_entry.get("id")
                node = stage_entry.get("node")
                rid = stage_entry.get("reservation_id")
                if not stage_id or not node or not rid:
                    continue

                stage_cfg = stage_by_id.get(stage_id) or {}
                resources = stage_cfg.get("resources") or {}
                managed = ManagedStage(
                    job_id=job_id,
                    stage_id=stage_id,
                    resources={
                        "cpu_cores": safe_float(resources.get("cpu_cores"), 0.0),
                        "mem_gb": safe_float(resources.get("mem_gb"), 0.0),
                        "gpu_vram_gb": safe_float(resources.get("gpu_vram_gb"), 0.0),
                    },
                    primary_node=node,
                    reservation_id=rid,
                    fallbacks=list(stage_entry.get("fallbacks") or plan.get("shadow_assignments", {}).get(stage_id, [])),
                    redundancy=max(redundancy, 1),
                )

                shadows = []
                for shadow in stage_entry.get("shadow_reservations", []):
                    snode = shadow.get("node")
                    srid = shadow.get("reservation_id")
                    if snode and srid:
                        shadows.append(ShadowReservation(node=snode, reservation_id=srid))
                managed.shadow_reservations = shadows

                bucket[stage_id] = managed

    def forget_reservation(self, reservation_id: str) -> None:
        if not reservation_id:
            return
        with self._lock:
            to_delete: List[tuple[str, str]] = []
            for job_id, stages in self._managed.items():
                for stage_id, managed in stages.items():
                    if managed.reservation_id == reservation_id:
                        to_delete.append((job_id, stage_id))
                        continue
                    managed.shadow_reservations = [
                        sh for sh in managed.shadow_reservations if sh.reservation_id != reservation_id
                    ]
            for job_id, stage_id in to_delete:
                self._managed[job_id].pop(stage_id, None)
                if not self._managed[job_id]:
                    self._managed.pop(job_id, None)

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

    # ------------------------- internals -------------------------

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._run_once()
            except Exception:
                # Never let background errors crash the main process; they will be logged by callers.
                pass
            self._stop.wait(self.poll_interval)

    def _run_once(self) -> None:
        with self._lock:
            snapshot = [(job_id, dict(stages)) for job_id, stages in self._managed.items()]

        for job_id, stages in snapshot:
            for stage_id, managed in stages.items():
                node_state = self.state.get_node(managed.primary_node)
                if node_state is None:
                    degraded = True
                else:
                    dyn = node_state.get("dyn") or {}
                    degraded = bool(dyn.get("down"))
                    reliability = safe_float(dyn.get("reliability"), 1.0)
                    if reliability < self.reliability_threshold:
                        degraded = True
                    window = dyn.get("predicted_failure_window_sec")
                    if window is not None and safe_float(window, 0.0) < self.availability_threshold_sec:
                        degraded = True

                if degraded:
                    managed.degraded_ticks += 1
                else:
                    managed.degraded_ticks = 0
                    self._ensure_shadow_capacity(managed)
                    continue

                if managed.degraded_ticks < self.stability_window:
                    continue

                if self._promote_shadow(managed):
                    managed.degraded_ticks = 0
                    self._ensure_shadow_capacity(managed)
                    continue

                if self._reserve_new_primary(managed):
                    managed.degraded_ticks = 0
                    self._ensure_shadow_capacity(managed)

    def _promote_shadow(self, managed: ManagedStage) -> bool:
        # Prefer the freshest standby
        managed.shadow_reservations.sort(key=lambda sh: sh.created_ts)
        while managed.shadow_reservations:
            standby = managed.shadow_reservations.pop(0)
            node_state = self.state.get_node(standby.node)
            if not node_state:
                continue
            dyn = node_state.get("dyn") or {}
            if standby.reservation_id not in (dyn.get("reservations") or {}):
                continue

            if self.state.release(managed.primary_node, managed.reservation_id):
                managed.primary_node = standby.node
                managed.reservation_id = standby.reservation_id
                managed.last_transition = time.time()
                self.state._emit_event(  # type: ignore[attr-defined]
                    "fabric.selfheal.promote",
                    {
                        "job_id": managed.job_id,
                        "stage_id": managed.stage_id,
                        "node": standby.node,
                        "reservation_id": standby.reservation_id,
                    },
                    subject=standby.node,
                )
                return True
        return False

    def _reserve_new_primary(self, managed: ManagedStage) -> bool:
        candidates = list(managed.fallbacks)
        random.shuffle(candidates)
        for candidate in candidates:
            node_name = candidate.get("node") if isinstance(candidate, dict) else candidate
            if not node_name or node_name == managed.primary_node:
                continue
            req = {
                "node": node_name,
                "cpu_cores": managed.resources.get("cpu_cores", 0.0),
                "mem_gb": managed.resources.get("mem_gb", 0.0),
                "gpu_vram_gb": managed.resources.get("gpu_vram_gb", 0.0),
            }
            rid = self.state.reserve(req)
            if not rid:
                continue
            if self.state.release(managed.primary_node, managed.reservation_id):
                managed.primary_node = node_name
                managed.reservation_id = rid
                managed.last_transition = time.time()
                self.state._emit_event(  # type: ignore[attr-defined]
                    "fabric.selfheal.failover",
                    {
                        "job_id": managed.job_id,
                        "stage_id": managed.stage_id,
                        "node": node_name,
                        "reservation_id": rid,
                    },
                    subject=node_name,
                )
                return True
            else:
                # If we can't release the old reservation, return the new one.
                self.state.release(node_name, rid)
        return False

    def _ensure_shadow_capacity(self, managed: ManagedStage) -> None:
        desired_shadows = max(0, managed.redundancy - 1)
        current = len(managed.shadow_reservations)
        if current >= desired_shadows:
            return

        candidates = list(managed.fallbacks)
        random.shuffle(candidates)
        for candidate in candidates:
            if len(managed.shadow_reservations) >= desired_shadows:
                break
            node_name = candidate.get("node") if isinstance(candidate, dict) else candidate
            if not node_name or node_name in {managed.primary_node} | {sh.node for sh in managed.shadow_reservations}:
                continue
            req = {
                "node": node_name,
                "cpu_cores": managed.resources.get("cpu_cores", 0.0),
                "mem_gb": managed.resources.get("mem_gb", 0.0),
                "gpu_vram_gb": managed.resources.get("gpu_vram_gb", 0.0),
            }
            rid = self.state.reserve(req)
            if not rid:
                continue
            managed.shadow_reservations.append(ShadowReservation(node=node_name, reservation_id=rid))
            self.state._emit_event(  # type: ignore[attr-defined]
                "fabric.selfheal.shadow",
                {
                    "job_id": managed.job_id,
                    "stage_id": managed.stage_id,
                    "node": node_name,
                    "reservation_id": rid,
                },
                subject=node_name,
            )


class ResourceGuardian:
    """Watches for potential deadlocks and releases stale reservations."""

    def __init__(
        self,
        state: DTState,
        *,
        poll_interval: float = 5.0,
        utilization_threshold: float = 0.92,
        reservation_ttl_sec: float = 300.0,
    ) -> None:
        self.state = state
        self.poll_interval = max(1.0, poll_interval)
        self.utilization_threshold = utilization_threshold
        self.reservation_ttl_sec = reservation_ttl_sec

        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="ResourceGuardian", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._inspect()
            except Exception:
                pass
            self._stop.wait(self.poll_interval)

    def _inspect(self) -> None:
        now = time.time()
        reservations = self.state.reservations_view()
        for node_name, res_map in reservations.items():
            headroom = self.state.node_headroom(node_name)
            if not headroom:
                continue

            max_cpu = safe_float(headroom.get("max_cpu_cores"), 0.0)
            free_cpu = safe_float(headroom.get("free_cpu_cores"), max_cpu)
            if max_cpu <= 0:
                continue

            utilization = 1.0 - (free_cpu / max_cpu)
            if utilization < self.utilization_threshold:
                continue

            oldest = None
            for rid, info in res_map.items():
                created_ts = safe_float(info.get("ts"), 0.0) / 1000.0
                age = now - created_ts if created_ts else float("inf")
                if age >= self.reservation_ttl_sec:
                    if oldest is None or age > oldest[0]:
                        oldest = (age, rid)
            if oldest is None:
                continue

            _, rid = oldest
            if self.state.release(node_name, rid):
                self.state._emit_event(  # type: ignore[attr-defined]
                    "fabric.guardian.release",
                    {
                        "node": node_name,
                        "reservation_id": rid,
                        "reason": "ttl_exceeded",
                    },
                    subject=node_name,
                )

