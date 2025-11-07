#!/usr/bin/env python3
"""
Validate ./nodes/*.yaml against schemas/node.schema.yaml.

Usage:
  python3 tools/validate_nodes.py
  python3 tools/validate_nodes.py --dir nodes --schema schemas/node.schema.yaml --strict
  python3 tools/validate_nodes.py --dir nodes --fail-fast

Features:
- Draft 2020-12 JSON Schema validation.
- Human-friendly error printing with JSON Pointer to the offending field.
- --strict: warns on common issues (missing accelerator hints, empty labels, etc.).
- --fail-fast: stop at first invalid file.
- Prints a summary table and returns non-zero on any validation error.

Note: JSON Schema "default" does not modify instances by default. This script
can optionally apply defaults if you pass --apply-defaults.
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml
from jsonschema import Draft202012Validator, RefResolver, validators, exceptions as js_ex

# --------------------------
# Helpers
# --------------------------

def load_yaml(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def human_path(p: Path, root: Path) -> str:
    try:
        return p.relative_to(root).as_posix()
    except Exception:
        return p.as_posix()

def extend_with_default(validator_class):
    """Make validator that sets defaults onto the instance (opt-in)."""
    validate_props = validator_class.VALIDATORS["properties"]
    def set_defaults(validator, properties, instance, schema):
        for prop, subschema in (properties or {}).items():
            if "default" in subschema:
                instance.setdefault(prop, subschema["default"])
        for error in validate_props(validator, properties, instance, schema):
            yield error
    return validators.extend(validator_class, {"properties": set_defaults})

def format_error(err: js_ex.ValidationError, file_rel: str) -> str:
    loc = "/".join([str(x) for x in err.path]) if err.path else "(root)"
    sch = "/".join([str(x) for x in err.schema_path])
    return (
        f"File: {file_rel}\n"
        f"  At:   $.{loc}\n"
        f"  Rule: {sch}\n"
        f"  Msg:  {err.message}"
    )

def strict_warnings(node: Dict[str, Any]) -> List[str]:
    """Non-fatal suggestions to improve simulation fidelity."""
    w: List[str] = []
    name = node.get("name", "<unnamed>")
    labels = node.get("labels", {})
    net = node.get("network", {})
    cpu = node.get("cpu", {})
    gpu = node.get("gpu", {})
    acc = node.get("accelerators", {})

    # Trust label present & parseable
    t = labels.get("trust")
    try:
        if t is None or not (0.0 <= float(t) <= 1.0):
            w.append(f"{name}: labels.trust missing or not in [0,1].")
    except Exception:
        w.append(f"{name}: labels.trust not a float-like string.")

    # Network sanity
    if net and net.get("speed_gbps", 0) <= 0:
        w.append(f"{name}: network.speed_gbps <= 0.")
    if net and net.get("base_bandwidth_mbps") and net.get("speed_gbps"):
        # Optional consistency check
        mbps = float(net["speed_gbps"]) * 1000.0
        if abs(float(net["base_bandwidth_mbps"]) - mbps) / max(1.0, mbps) > 0.5:
            w.append(f"{name}: base_bandwidth_mbps differs wildly from speed_gbps*1000.")

    # CPU must have sensible turbo >= base
    try:
        if float(cpu.get("turbo_ghz", 0)) < float(cpu.get("base_ghz", 0)):
            w.append(f"{name}: cpu.turbo_ghz < cpu.base_ghz.")
    except Exception:
        pass

    # Accelerator hints
    fmts = set(node.get("formats_supported") or [])
    if "cuda" in fmts and (gpu.get("type") == "none" or float(gpu.get("accel_score", 0)) <= 0):
        w.append(f"{name}: formats_supported includes 'cuda' but GPU block looks empty.")
    if "npu" in fmts and (not acc or acc.get("npu","none") == "none"):
        w.append(f"{name}: formats_supported includes 'npu' but accelerators.npu is 'none'.")

    # Storage health
    st = node.get("storage", {})
    if st:
        if float(st.get("tbw_pct_used", 0)) > 95:
            w.append(f"{name}: storage.tbw_pct_used > 95% (high wear).")

    return w

# --------------------------
# Main validation
# --------------------------

def validate_nodes(
    nodes_dir: Path,
    schema_path: Path,
    apply_defaults: bool = False,
    strict: bool = False,
    fail_fast: bool = False,
) -> Tuple[int, int, int]:
    """
    Returns: (num_files, num_invalid, num_warn)
    """
    schema = load_yaml(schema_path)
    base_uri = "file://" + str(schema_path.resolve())
    resolver = RefResolver(base_uri=base_uri, referrer=schema)

    ValidatorClass = Draft202012Validator
    if apply_defaults:
        ValidatorClass = extend_with_default(Draft202012Validator)

    validator = ValidatorClass(schema, resolver=resolver)

    files = sorted(nodes_dir.glob("*.yaml"))
    if not files:
        print(f"[error] No YAML files found in {nodes_dir.as_posix()}")
        return (0, 0, 0)

    invalid = 0
    warned = 0
    root = Path.cwd()

    for f in files:
        inst = load_yaml(f)
        errs = sorted(validator.iter_errors(inst), key=lambda e: e.path)
        if errs:
            invalid += 1
            print("=" * 80)
            print(format_error(errs[0], human_path(f, root)))
            # Print additional errors (up to a few) for context
            for e in errs[1:5]:
                print("- " + e.message)
            if fail_fast:
                break
        else:
            if strict:
                warns = strict_warnings(inst)
                if warns:
                    warned += len(warns)
                    print("-" * 80)
                    print(f"File: {human_path(f, root)}  (valid, {len(warns)} warning(s))")
                    for w in warns:
                        print(f"  warn: {w}")

    # Summary
    print("\nSummary")
    print("-------")
    print(f"Directory : {nodes_dir.as_posix()}")
    print(f"Schema    : {schema_path.as_posix()}")
    print(f"Checked   : {len(files)} file(s)")
    print(f"Invalid   : {invalid}")
    print(f"Warnings  : {warned} (strict={'on' if strict else 'off'})")

    return (len(files), invalid, warned)

# --------------------------
# CLI
# --------------------------

def main():
    ap = argparse.ArgumentParser(description="Validate node descriptors against node.schema.yaml")
    ap.add_argument("--dir", default="nodes", help="Directory with per-node YAMLs")
    ap.add_argument("--schema", default="schemas/node.schema.yaml", help="Path to node schema")
    ap.add_argument("--apply-defaults", action="store_true", help="Apply JSON Schema defaults before validating")
    ap.add_argument("--strict", action="store_true", help="Emit additional non-fatal warnings")
    ap.add_argument("--fail-fast", action="store_true", help="Stop at first invalid file")
    args = ap.parse_args()

    nodes_dir = Path(args.dir)
    schema_path = Path(args.schema)

    total, invalid, _ = validate_nodes(
        nodes_dir=nodes_dir,
        schema_path=schema_path,
        apply_defaults=args.apply_defaults,
        strict=args.strict,
        fail_fast=args.fail_fast,
    )

    if total == 0 or invalid > 0:
        sys.exit(1)
    sys.exit(0)

if __name__ == "__main__":
    main()

