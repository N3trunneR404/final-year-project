#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dt/validators.py — JSON/YAML schema validation utilities for Fabric DT.

Features
--------
- Load & cache Draft 2020-12 JSON Schemas from ./schemas
- Validate single instances (node, job, topology) or a whole nodes/ dir
- Two modes:
    • lint_*()    → return list of (path, message) problems (non-throwing)
    • assert_*()  → raise ValidationError on first problem
- Optional default injection (schema "default" → instance), opt-in per call

Dependencies
------------
pip install jsonschema pyyaml

Conventions
-----------
Default schema filenames (configurable per call):
- schemas/node.schema.yaml
- schemas/job.schema.yaml
- schemas/topology.schema.yaml
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml
from jsonschema import Draft202012Validator, RefResolver, validators, exceptions as js_ex


# --------------------------- Exceptions ---------------------------

class ValidationError(RuntimeError):
    def __init__(self, where: str, message: str, schema_path: str = "", instance_path: str = ""):
        super().__init__(f"{where}: {message} (at $.{instance_path}; rule {schema_path})")
        self.where = where
        self.message = message
        self.schema_path = schema_path
        self.instance_path = instance_path


# --------------------------- Helpers ------------------------------

def _extend_with_default(validator_class):
    """Return a validator that sets defaults onto instances (opt-in)."""
    validate_props = validator_class.VALIDATORS["properties"]

    def set_defaults(validator, properties, instance, schema):
        if isinstance(instance, dict):
            for prop, subschema in (properties or {}).items():
                if "default" in subschema and prop not in instance:
                    instance[prop] = subschema["default"]
        for error in validate_props(validator, properties, instance, schema):
            yield error

    return validators.extend(validator_class, {"properties": set_defaults})


def _format_error(err: js_ex.ValidationError) -> Tuple[str, str]:
    """Return (instance_pointer, schema_pointer) strings."""
    inst = "/".join([str(x) for x in err.path]) if err.path else "(root)"
    sch = "/".join([str(x) for x in err.schema_path])
    return inst, sch


def _load_yaml(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# --------------------------- Schema Cache -------------------------

@dataclass
class _CompiledSchema:
    path: Path
    validator: Draft202012Validator


class SchemaRegistry:
    """
    Load and cache JSON Schemas from a directory. Handles $ref via file:// URIs.
    """
    def __init__(self, schemas_dir: str = "schemas"):
        self.schemas_dir = Path(schemas_dir)
        self._cache: Dict[str, _CompiledSchema] = {}
        self._default_cls = Draft202012Validator
        self._default_with_defs = _extend_with_default(Draft202012Validator)

    def _compile(self, name: str, apply_defaults: bool = False) -> _CompiledSchema:
        path = (self.schemas_dir / name).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Schema not found: {path}")
        raw = _load_yaml(path)
        base_uri = "file://" + str(path)
        resolver = RefResolver(base_uri=base_uri, referrer=raw)
        cls = self._default_with_defs if apply_defaults else self._default_cls
        validator = cls(raw, resolver=resolver)
        return _CompiledSchema(path=path, validator=validator)

    def get(self, name: str, apply_defaults: bool = False) -> _CompiledSchema:
        key = f"{name}::defaults={bool(apply_defaults)}"
        c = self._cache.get(key)
        if c is None:
            c = self._compile(name, apply_defaults=apply_defaults)
            self._cache[key] = c
        return c


# --------------------------- Public API ---------------------------

DEFAULT_NODE_SCHEMA = "node.schema.yaml"
DEFAULT_JOB_SCHEMA = "job.schema.yaml"
DEFAULT_TOPOLOGY_SCHEMA = "topology.schema.yaml"


def lint_instance(
    instance: Any,
    schema_file: str,
    registry: Optional[SchemaRegistry] = None,
    apply_defaults: bool = False,
) -> List[Tuple[str, str]]:
    """
    Validate any instance against the given schema file.
    Returns a list of (instance_pointer, message).
    """
    reg = registry or SchemaRegistry()
    compiled = reg.get(schema_file, apply_defaults=apply_defaults)
    errs = sorted(compiled.validator.iter_errors(instance), key=lambda e: e.path)
    problems: List[Tuple[str, str]] = []
    for e in errs:
        inst_ptr, _ = _format_error(e)
        problems.append((inst_ptr, e.message))
    return problems


def assert_instance(
    instance: Any,
    schema_file: str,
    where: str = "instance",
    registry: Optional[SchemaRegistry] = None,
    apply_defaults: bool = False,
) -> None:
    """
    Validate and raise ValidationError on the first problem.
    """
    reg = registry or SchemaRegistry()
    compiled = reg.get(schema_file, apply_defaults=apply_defaults)
    for e in compiled.validator.iter_errors(instance):
        inst_ptr, sch_ptr = _format_error(e)
        raise ValidationError(where=where, message=e.message, schema_path=sch_ptr, instance_path=inst_ptr)


# ---- Specific convenience wrappers ----

def lint_node(node: Dict[str, Any], registry: Optional[SchemaRegistry] = None, apply_defaults: bool = False):
    return lint_instance(node, DEFAULT_NODE_SCHEMA, registry=registry, apply_defaults=apply_defaults)

def assert_node(node: Dict[str, Any], registry: Optional[SchemaRegistry] = None, apply_defaults: bool = False):
    return assert_instance(node, DEFAULT_NODE_SCHEMA, where=node.get("name", "node"), registry=registry, apply_defaults=apply_defaults)

def lint_job(job: Dict[str, Any], registry: Optional[SchemaRegistry] = None, apply_defaults: bool = False):
    return lint_instance(job, DEFAULT_JOB_SCHEMA, registry=registry, apply_defaults=apply_defaults)

def assert_job(job: Dict[str, Any], registry: Optional[SchemaRegistry] = None, apply_defaults: bool = False):
    return assert_instance(job, DEFAULT_JOB_SCHEMA, where=job.get("id", "job"), registry=registry, apply_defaults=apply_defaults)

def lint_topology(topology: Dict[str, Any], registry: Optional[SchemaRegistry] = None, apply_defaults: bool = False):
    return lint_instance(topology, DEFAULT_TOPOLOGY_SCHEMA, registry=registry, apply_defaults=apply_defaults)

def assert_topology(topology: Dict[str, Any], registry: Optional[SchemaRegistry] = None, apply_defaults: bool = False):
    return assert_instance(topology, DEFAULT_TOPOLOGY_SCHEMA, where="topology", registry=registry, apply_defaults=apply_defaults)


# ---- Directory helpers ----

@dataclass
class DirReport:
    total: int
    valid: int
    invalid: int
    files: List[Dict[str, Any]]  # [{path, valid, problems:[(ptr,msg),...]}]

def validate_nodes_dir(
    nodes_dir: str = "nodes",
    registry: Optional[SchemaRegistry] = None,
    apply_defaults: bool = False,
) -> DirReport:
    """
    Validate all *.yaml in a directory against node.schema.yaml.
    Returns a DirReport with per-file results.
    """
    reg = registry or SchemaRegistry()
    compiled = reg.get(DEFAULT_NODE_SCHEMA, apply_defaults=apply_defaults)

    base = Path(nodes_dir)
    results: List[Dict[str, Any]] = []
    valid = 0
    invalid = 0
    files = sorted(base.glob("*.yaml"))

    for f in files:
        try:
            inst = _load_yaml(f)
        except Exception as e:
            invalid += 1
            results.append({"path": f.as_posix(), "valid": False, "problems": [("(root)", f"failed to load YAML: {e}")]})
            continue

        errs = sorted(compiled.validator.iter_errors(inst), key=lambda e: e.path)
        if errs:
            invalid += 1
            probs = []
            for e in errs:
                ptr, _ = _format_error(e)
                probs.append((ptr, e.message))
            results.append({"path": f.as_posix(), "valid": False, "problems": probs})
        else:
            valid += 1
            results.append({"path": f.as_posix(), "valid": True, "problems": []})

    return DirReport(total=len(files), valid=valid, invalid=invalid, files=results)


# --------------------------- CLI (optional) ------------------------

if __name__ == "__main__":
    import argparse, json, sys

    ap = argparse.ArgumentParser(description="Validate Fabric DT YAMLs")
    ap.add_argument("--schemas", default="schemas", help="Schemas directory")
    ap.add_argument("--nodes", default=None, help="Validate all nodes in directory")
    ap.add_argument("--job", default=None, help="Validate a single job YAML")
    ap.add_argument("--topology", default=None, help="Validate topology YAML")
    ap.add_argument("--apply-defaults", action="store_true", help="Apply schema defaults to instances")
    args = ap.parse_args()

    reg = SchemaRegistry(args.schemas)

    try:
        if args.nodes:
            rep = validate_nodes_dir(args.nodes, registry=reg, apply_defaults=args.apply_defaults)
            print(json.dumps({
                "total": rep.total,
                "valid": rep.valid,
                "invalid": rep.invalid,
                "files": rep.files
            }, indent=2))
            sys.exit(0 if rep.invalid == 0 and rep.total > 0 else 1)

        if args.job:
            job = _load_yaml(Path(args.job))
            probs = lint_job(job, registry=reg, apply_defaults=args.apply_defaults)
            if probs:
                print(json.dumps({"ok": False, "problems": probs}, indent=2))
                sys.exit(1)
            print(json.dumps({"ok": True}, indent=2))
            sys.exit(0)

        if args.topology:
            topo = _load_yaml(Path(args.topology))
            probs = lint_topology(topo, registry=reg, apply_defaults=args.apply_defaults)
            if probs:
                print(json.dumps({"ok": False, "problems": probs}, indent=2))
                sys.exit(1)
            print(json.dumps({"ok": True}, indent=2))
            sys.exit(0)

        ap.print_help()
        sys.exit(2)
    except ValidationError as ve:
        print(str(ve))
        sys.exit(1)
    except Exception as e:
        print(f"fatal: {e}")
        sys.exit(1)

