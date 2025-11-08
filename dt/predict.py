#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Helpers for predictive telemetry and trend-aware scoring.

The predictors here remain intentionally lightweight so they can execute inside
unit tests and notebooks without external dependencies.  They provide a small
set of exponential moving averages and linear-trend estimators that the digital
"twin" can use to reason about near-future load, availability, and
reliability.

The :class:`PredictiveAnalyzer` aggregates telemetry for both nodes and links
and exposes convenience accessors used by :mod:`dt.state` and the cost model.

The estimators assume inputs expressed in SI units (seconds, watts, etc.) and
expect callers to supply timestamps in milliseconds since the UNIX epoch.  The
APIs fall back to monotonic counters when timestamps are omitted so that unit
and integration tests can continue to provide deterministic data.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Optional

def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _now_ms() -> float:
    return time.time() * 1000.0


@dataclass
class _EWMA:
    """Exponentially-weighted moving average."""

    alpha: float
    value: Optional[float] = None

    def update(self, sample: float) -> float:
        if self.value is None:
            self.value = sample
        else:
            self.value = self.alpha * sample + (1.0 - self.alpha) * self.value
        return self.value


@dataclass
class _Trend:
    """Small helper that tracks a rolling trend via least squares."""

    window: int
    count: int = 0
    sum_x: float = 0.0
    sum_y: float = 0.0
    sum_xx: float = 0.0
    sum_xy: float = 0.0

    def update(self, sample: float, timestamp_ms: Optional[float] = None) -> float:
        ts = timestamp_ms if timestamp_ms is not None else self.count
        # Forget the previous contribution when the window is exceeded by
        # decaying the accumulated sums.  This avoids retaining full histories
        # and keeps the implementation constant memory.
        decay = 1.0
        if self.count >= self.window and self.window > 0:
            decay = max(0.0, float(self.window - 1) / float(self.window))
        self.sum_x *= decay
        self.sum_y *= decay
        self.sum_xx *= decay
        self.sum_xy *= decay

        self.sum_x += ts
        self.sum_y += sample
        self.sum_xx += ts * ts
        self.sum_xy += ts * sample
        self.count = min(self.count + 1, max(self.window, self.count + 1))

        denom = self.sum_xx * self.count - self.sum_x * self.sum_x
        if abs(denom) < 1e-9:
            return 0.0
        slope = (self.count * self.sum_xy - self.sum_x * self.sum_y) / denom
        return slope


@dataclass
class NodeForecast:
    util_now: float = 0.0
    util_forecast: float = 0.0
    reliability: float = 1.0
    availability_window_sec: Optional[float] = None
    projected_derate: float = 0.0


@dataclass
class LinkForecast:
    latency_ms: float = 0.0
    jitter_ms: float = 0.0
    loss_pct: float = 0.0
    latency_p95_ms: Optional[float] = None


class PredictiveAnalyzer:
    """Aggregate predictive telemetry for nodes and links."""

    def __init__(self, *, util_alpha: float = 0.35, window: int = 12):
        self._util_alpha = util_alpha
        self._window = max(3, window)
        self._node_util: Dict[str, _EWMA] = {}
        self._node_trend: Dict[str, _Trend] = {}
        self._node_derate: Dict[str, _EWMA] = {}
        self._node_reliability: Dict[str, float] = {}
        self._node_availability: Dict[str, Optional[float]] = {}
        self._node_last_ts: Dict[str, float] = {}
        self._node_battery_pct: Dict[str, Optional[float]] = {}
        self._node_battery_drain: Dict[str, Optional[float]] = {}
        self._node_mtbf: Dict[str, Optional[float]] = {}
        self._node_uptime: Dict[str, Optional[float]] = {}

        self._link_latency: Dict[str, _EWMA] = {}
        self._link_jitter: Dict[str, _EWMA] = {}
        self._link_loss: Dict[str, _EWMA] = {}
        self._link_trend: Dict[str, _Trend] = {}

    # ---- registration -------------------------------------------------

    def ensure_node(
        self,
        name: str,
        *,
        reliability: Optional[float] = None,
        availability_window_sec: Optional[float] = None,
        battery_pct: Optional[float] = None,
        battery_drain_pct_per_hr: Optional[float] = None,
        mtbf_hours: Optional[float] = None,
        uptime_hours: Optional[float] = None,
    ) -> None:
        self._node_util.setdefault(name, _EWMA(self._util_alpha))
        self._node_trend.setdefault(name, _Trend(self._window))
        self._node_derate.setdefault(name, _EWMA(self._util_alpha))
        if reliability is not None:
            self._node_reliability[name] = _clamp(float(reliability), 0.0, 1.0)
        self._node_availability.setdefault(name, availability_window_sec)
        self._node_battery_pct.setdefault(name, battery_pct)
        self._node_battery_drain.setdefault(name, battery_drain_pct_per_hr)
        self._node_mtbf.setdefault(name, mtbf_hours)
        self._node_uptime.setdefault(name, uptime_hours)

    def ensure_link(self, key: str) -> None:
        self._link_latency.setdefault(key, _EWMA(self._util_alpha))
        self._link_jitter.setdefault(key, _EWMA(self._util_alpha))
        self._link_loss.setdefault(key, _EWMA(self._util_alpha))
        self._link_trend.setdefault(key, _Trend(self._window))

    # ---- updates ------------------------------------------------------

    def record_node_util(
        self,
        name: str,
        util: float,
        *,
        thermal_derate: float = 0.0,
        reliability: Optional[float] = None,
        availability_window_sec: Optional[float] = None,
        battery_pct: Optional[float] = None,
        battery_drain_pct_per_hr: Optional[float] = None,
        mtbf_hours: Optional[float] = None,
        uptime_hours: Optional[float] = None,
        timestamp_ms: Optional[float] = None,
    ) -> NodeForecast:
        self.ensure_node(
            name,
            reliability=reliability,
            availability_window_sec=availability_window_sec,
            battery_pct=battery_pct,
            battery_drain_pct_per_hr=battery_drain_pct_per_hr,
            mtbf_hours=mtbf_hours,
            uptime_hours=uptime_hours,
        )
        ts = timestamp_ms if timestamp_ms is not None else _now_ms()
        util = _clamp(util, 0.0, 1.0)
        util_now = self._node_util[name].update(util)
        util_slope = self._node_trend[name].update(util_now, ts)
        derate = _clamp(thermal_derate, 0.0, 1.0)
        derate_forecast = self._node_derate[name].update(derate + max(0.0, util_slope))

        if reliability is not None:
            self._node_reliability[name] = _clamp(float(reliability), 0.0, 1.0)
        if availability_window_sec is not None:
            self._node_availability[name] = max(0.0, availability_window_sec)
        if battery_pct is not None:
            self._node_battery_pct[name] = _clamp(float(battery_pct), 0.0, 1.0)
        if battery_drain_pct_per_hr is not None:
            self._node_battery_drain[name] = float(battery_drain_pct_per_hr)
        if mtbf_hours is not None:
            self._node_mtbf[name] = max(0.0, float(mtbf_hours))
        if uptime_hours is not None:
            self._node_uptime[name] = max(0.0, float(uptime_hours))

        self._node_last_ts[name] = ts

        availability = self._availability_window(name)
        reliability_val = self._reliability(name)
        projected_derate = _clamp(derate_forecast, 0.0, 1.0)
        util_forecast = _clamp(util_now + util_slope * 30.0, 0.0, 1.0)

        return NodeForecast(
            util_now=util_now,
            util_forecast=util_forecast,
            reliability=reliability_val,
            availability_window_sec=availability,
            projected_derate=projected_derate,
        )

    def record_link_metrics(
        self,
        key: str,
        *,
        latency_ms: float,
        jitter_ms: float,
        loss_pct: float,
        timestamp_ms: Optional[float] = None,
    ) -> LinkForecast:
        self.ensure_link(key)
        ts = timestamp_ms if timestamp_ms is not None else _now_ms()
        lat = self._link_latency[key].update(max(0.0, float(latency_ms)))
        jit = self._link_jitter[key].update(max(0.0, float(jitter_ms)))
        loss = self._link_loss[key].update(_clamp(float(loss_pct), 0.0, 100.0))
        slope = self._link_trend[key].update(lat, ts)

        # Approximate a P95 using the EWMA and trend slope.  This is not a
        # statistical guarantee but provides a conservative estimate for
        # scheduling decisions.
        p95 = lat + abs(slope) * 4.0 + jit * 2.0
        return LinkForecast(
            latency_ms=lat,
            jitter_ms=jit,
            loss_pct=loss,
            latency_p95_ms=p95,
        )

    # ---- accessors ----------------------------------------------------

    def _reliability(self, name: str) -> float:
        base = self._node_reliability.get(name, 0.95)
        mtbf = self._node_mtbf.get(name)
        uptime = self._node_uptime.get(name)
        if mtbf and uptime:
            failure_prob = min(1.0, max(0.0, uptime / max(mtbf, 1e-3)))
            base *= (1.0 - 0.5 * failure_prob)
        batt_pct = self._node_battery_pct.get(name)
        drain = self._node_battery_drain.get(name)
        if batt_pct is not None and drain is not None and drain > 0:
            hours_left = max(0.0, batt_pct * 100.0 / max(drain, 1e-6))
            if hours_left < 1.0:
                base *= 0.5
            elif hours_left < 2.0:
                base *= 0.7
        return _clamp(base, 0.0, 1.0)

    def _availability_window(self, name: str) -> Optional[float]:
        avail = self._node_availability.get(name)
        if avail is not None:
            return max(0.0, float(avail))
        batt_pct = self._node_battery_pct.get(name)
        drain = self._node_battery_drain.get(name)
        if batt_pct is not None and drain is not None and drain > 0:
            hours_left = max(0.0, batt_pct * 100.0 / max(drain, 1e-6))
            return hours_left * 3600.0
        return None

    def node_forecast(self, name: str) -> NodeForecast:
        util_now = self._node_util.get(name, _EWMA(self._util_alpha)).value or 0.0
        derate = self._node_derate.get(name, _EWMA(self._util_alpha)).value or 0.0
        return NodeForecast(
            util_now=util_now,
            util_forecast=util_now,
            reliability=self._reliability(name),
            availability_window_sec=self._availability_window(name),
            projected_derate=_clamp(derate, 0.0, 1.0),
        )

    def link_forecast(self, key: str) -> LinkForecast:
        latency = self._link_latency.get(key)
        jitter = self._link_jitter.get(key)
        loss = self._link_loss.get(key)
        trend = self._link_trend.get(key)
        lat = latency.value if latency else 0.0
        jit = jitter.value if jitter else 0.0
        lossv = loss.value if loss else 0.0
        slope = trend.sum_xy if trend else 0.0
        p95 = lat + abs(slope) * 4.0 + jit * 2.0
        return LinkForecast(latency_ms=lat, jitter_ms=jit, loss_pct=lossv, latency_p95_ms=p95)

    # ---- summaries ----------------------------------------------------

    def overview(self) -> Dict[str, Dict[str, float]]:
        summary: Dict[str, Dict[str, float]] = {"nodes": {}, "links": {}}
        for name in self._node_util.keys():
            fc = self.node_forecast(name)
            summary["nodes"][name] = {
                "util_now": round(fc.util_now, 4),
                "util_forecast": round(fc.util_forecast, 4),
                "reliability": round(fc.reliability, 4),
                "availability_window_sec": None
                if fc.availability_window_sec is None
                else round(fc.availability_window_sec, 3),
                "projected_derate": round(fc.projected_derate, 4),
            }
        for key in self._link_latency.keys():
            lc = self.link_forecast(key)
            summary["links"][key] = {
                "latency_ms": round(lc.latency_ms, 3),
                "jitter_ms": round(lc.jitter_ms, 3),
                "loss_pct": round(lc.loss_pct, 4),
                "latency_p95_ms": None
                if lc.latency_p95_ms is None
                else round(lc.latency_p95_ms, 3),
            }
        return summary

