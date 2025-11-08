#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export helpers for open standards (DTDL, Kubernetes CRDs)."""
from __future__ import annotations

from typing import Any, Dict, List

from .state import DTState


def as_dtdl(state: DTState) -> Dict[str, Any]:
    snap = state.snapshot()
    models: List[Dict[str, Any]] = []
    for node in snap.get("nodes", []):
        name = node.get("name")
        if not name:
            continue
        ident = f"dtmi:fabric:node:{name};1"
        dyn = node.get("dyn") or {}
        effective = node.get("effective") or {}
        models.append(
            {
                "@id": ident,
                "@type": "Interface",
                "displayName": name,
                "contents": [
                    {"@type": "Property", "name": "class", "schema": "string"},
                    {"@type": "Property", "name": "arch", "schema": "string"},
                    {"@type": "Property", "name": "labels", "schema": {"@type": "Map", "mapKey": {"name": "key", "schema": "string"}, "mapValue": {"name": "value", "schema": "string"}}},
                    {"@type": "Property", "name": "formatsSupported", "schema": {"@type": "Array", "elementSchema": "string"}},
                    {"@type": "Telemetry", "name": "cpuUtil", "schema": "double", "description": "Current CPU utilisation fraction"},
                    {"@type": "Telemetry", "name": "memUtil", "schema": "double"},
                    {"@type": "Telemetry", "name": "gpuUtil", "schema": "double"},
                    {"@type": "Telemetry", "name": "reliability", "schema": "double", "description": "Predictive reliability score"},
                    {"@type": "Property", "name": "availableCpuCores", "schema": "double", "displayName": "Free CPU cores", "value": effective.get("free_cpu_cores")},
                    {"@type": "Property", "name": "availableMemGb", "schema": "double", "value": effective.get("free_mem_gb")},
                ],
                "contentsExtra": dyn,
            }
        )
    for link in snap.get("links", []):
        key = link.get("key")
        if not key:
            continue
        ident = f"dtmi:fabric:link:{key.replace('|', ':')};1"
        eff = link.get("effective") or {}
        models.append(
            {
                "@id": ident,
                "@type": "Interface",
                "displayName": key,
                "contents": [
                    {"@type": "Property", "name": "endpoints", "schema": {"@type": "Array", "elementSchema": "string"}},
                    {"@type": "Telemetry", "name": "latencyMs", "schema": "double"},
                    {"@type": "Telemetry", "name": "jitterMs", "schema": "double"},
                    {"@type": "Telemetry", "name": "lossPct", "schema": "double"},
                    {"@type": "Telemetry", "name": "speedGbps", "schema": "double"},
                ],
                "contentsExtra": eff,
            }
        )
    return {"models": models}


def as_k8s_crds(state: DTState) -> Dict[str, Any]:
    snap = state.snapshot()
    node_instances: List[Dict[str, Any]] = []
    link_instances: List[Dict[str, Any]] = []
    for node in snap.get("nodes", []):
        name = node.get("name")
        if not name:
            continue
        node_instances.append(
            {
                "apiVersion": "fabric/v1alpha1",
                "kind": "NodeTwin",
                "metadata": {"name": name},
                "spec": {
                    "class": node.get("class"),
                    "arch": node.get("arch"),
                    "labels": node.get("labels", {}),
                    "formatsSupported": node.get("formats_supported", []),
                    "capacity": node.get("caps", {}),
                },
                "status": {
                    "dynamic": node.get("dyn", {}),
                    "effective": node.get("effective", {}),
                    "predictive": state.predictive_overview().get("nodes", {}).get(name, {}),
                },
            }
        )
    for link in snap.get("links", []):
        key = link.get("key")
        if not key:
            continue
        a = link.get("a")
        b = link.get("b")
        link_instances.append(
            {
                "apiVersion": "fabric/v1alpha1",
                "kind": "LinkTwin",
                "metadata": {"name": key.replace("|", "-")},
                "spec": {"a": a, "b": b},
                "status": {
                    "dynamic": link.get("dyn", {}),
                    "effective": link.get("effective", {}),
                    "predictive": state.predictive_overview().get("links", {}).get(key, {}),
                },
            }
        )
    crds = {
        "definitions": {
            "NodeTwin": {
                "apiVersion": "apiextensions.k8s.io/v1",
                "kind": "CustomResourceDefinition",
                "metadata": {"name": "nodetwins.fabric.dev"},
                "spec": {
                    "group": "fabric.dev",
                    "scope": "Namespaced",
                    "names": {
                        "plural": "nodetwins",
                        "singular": "nodetwin",
                        "kind": "NodeTwin",
                        "shortNames": ["ntwin"],
                    },
                    "versions": [
                        {
                            "name": "v1alpha1",
                            "served": True,
                            "storage": True,
                            "schema": {"openAPIV3Schema": {"type": "object"}},
                        }
                    ],
                },
            },
            "LinkTwin": {
                "apiVersion": "apiextensions.k8s.io/v1",
                "kind": "CustomResourceDefinition",
                "metadata": {"name": "linktwins.fabric.dev"},
                "spec": {
                    "group": "fabric.dev",
                    "scope": "Namespaced",
                    "names": {
                        "plural": "linktwins",
                        "singular": "linktwin",
                        "kind": "LinkTwin",
                        "shortNames": ["ltwin"],
                    },
                    "versions": [
                        {
                            "name": "v1alpha1",
                            "served": True,
                            "storage": True,
                            "schema": {"openAPIV3Schema": {"type": "object"}},
                        }
                    ],
                },
            },
        },
        "instances": {"NodeTwin": node_instances, "LinkTwin": link_instances},
    }
    return crds

