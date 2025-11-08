#!/usr/bin/env python3
"""Automated noise injector to keep the twin lively."""

from __future__ import annotations

import random
import threading
import time
from typing import Dict, Optional

from .state import DTState


class NoiseInjector:
    def __init__(
        self,
        state: DTState,
        *,
        interval_sec: float = 7.5,
        node_jitter: float = 0.1,
        link_jitter: float = 0.2,
    ) -> None:
        self.state = state
        self.interval_sec = max(1.0, interval_sec)
        self.node_jitter = max(0.0, node_jitter)
        self.link_jitter = max(0.0, link_jitter)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="NoiseInjector", daemon=True)

    def start(self) -> None:
        if self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="NoiseInjector", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._inject()
            except Exception:
                pass
            self._stop.wait(self.interval_sec)

    def _inject(self) -> None:
        snapshot = self.state.snapshot()
        if not snapshot.get("nodes"):
            return
        node = random.choice(snapshot["nodes"])
        node_changes: Dict[str, float] = {}
        dyn = node.get("dyn") or {}
        if self.node_jitter > 0:
            thermal = float(dyn.get("thermal_derate", 0.0))
            thermal = max(0.0, min(0.95, thermal + random.uniform(-0.02, 0.05) * self.node_jitter))
            node_changes["thermal_derate"] = round(thermal, 4)
            used = float(dyn.get("used_cpu_cores", 0.0))
            node_changes["used_cpu_cores"] = max(0.0, used * random.uniform(0.9, 1.1))
            battery = dyn.get("battery_pct")
            if battery is not None:
                node_changes["battery_pct"] = max(0.0, min(100.0, float(battery) - random.uniform(0.1, 0.7)))

        self.state.apply_observation(
            {
                "payload": {
                    "type": "node",
                    "node": node.get("name"),
                    "changes": node_changes,
                }
            }
        )

        links = snapshot.get("links") or []
        if not links:
            return
        link = random.choice(links)
        link_changes: Dict[str, float] = {}
        if self.link_jitter > 0:
            rtt = float((link.get("dyn") or {}).get("rtt_ms") or (link.get("effective") or {}).get("rtt_ms", 5.0))
            jitter = float((link.get("dyn") or {}).get("jitter_ms") or (link.get("effective") or {}).get("jitter_ms", 0.5))
            loss = float((link.get("dyn") or {}).get("loss_pct") or (link.get("effective") or {}).get("loss_pct", 0.0))
            link_changes["rtt_ms"] = max(0.5, rtt * random.uniform(0.8, 1.3))
            link_changes["jitter_ms"] = max(0.1, jitter * random.uniform(0.5, 1.8))
            link_changes["loss_pct"] = max(0.0, min(15.0, loss + random.uniform(-0.2, 0.5)))

        self.state.apply_observation(
            {
                "payload": {
                    "type": "link",
                    "key": link.get("key"),
                    "changes": link_changes,
                }
            }
        )


def maybe_start_noise(state: DTState) -> Optional[NoiseInjector]:
    """Utility helper that honours FABRIC_ENABLE_NOISE."""

    import os

    enabled = os.environ.get("FABRIC_ENABLE_NOISE", "0").lower() in {"1", "true", "yes"}
    if not enabled:
        return None

    interval = float(os.environ.get("FABRIC_NOISE_INTERVAL", "7.5"))
    node_jitter = float(os.environ.get("FABRIC_NOISE_NODE_JITTER", "0.12"))
    link_jitter = float(os.environ.get("FABRIC_NOISE_LINK_JITTER", "0.2"))

    injector = NoiseInjector(
        state,
        interval_sec=interval,
        node_jitter=node_jitter,
        link_jitter=link_jitter,
    )
    injector.start()
    return injector

