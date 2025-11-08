#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Simple CloudEvents helpers for the Fabric Digital Twin."""
from __future__ import annotations

import threading
import time
import uuid
from collections import deque
from typing import Any, Deque, Dict, Iterable, Optional


def build_cloudevent(
    event_type: str,
    source: str,
    data: Dict[str, Any],
    *,
    subject: Optional[str] = None,
    time_ms: Optional[float] = None,
    event_id: Optional[str] = None,
    extensions: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    ts = (time_ms if time_ms is not None else time.time() * 1000.0) / 1000.0
    evt: Dict[str, Any] = {
        "specversion": "1.0",
        "id": event_id or str(uuid.uuid4()),
        "type": event_type,
        "source": source,
        "time": time.strftime("%Y-%m-%dT%H:%M:%S.%fZ", time.gmtime(ts)),
        "data": data,
    }
    if subject:
        evt["subject"] = subject
    if extensions:
        for key, value in extensions.items():
            evt[key] = value
    return evt


class EventBus:
    """Thread-safe in-memory event buffer."""

    def __init__(self, maxlen: int = 256):
        self._events: Deque[Dict[str, Any]] = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def emit(self, event: Dict[str, Any]) -> None:
        with self._lock:
            self._events.append(event)

    def recent(self, *, limit: int = 50, since_id: Optional[str] = None) -> Iterable[Dict[str, Any]]:
        with self._lock:
            items = list(self._events)[-limit:]
        if since_id is None:
            return items
        seen = False
        filtered = []
        for evt in items:
            if evt.get("id") == since_id:
                seen = True
                continue
            if not seen:
                continue
            filtered.append(evt)
        return filtered

