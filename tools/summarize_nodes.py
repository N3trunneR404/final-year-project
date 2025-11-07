#!/usr/bin/env python3
"""
Summarize per-node YAML descriptors (./nodes/*.yaml).

Examples
--------
# Basic console summary
python3 tools/summarize_nodes.py

# Choose dir, export CSV + Markdown report
python3 tools/summarize_nodes.py --dir nodes --csv sim/nodes_inventory.csv --md sim/nodes_report.md

# JSON dump for other tooling
python3 tools/summarize_nodes.py --json sim/nodes_summary.json

What it reports
---------------
- Totals by class (phone/sbc/laptop/workstation/gaming_rig/server/hpc)
- Totals by arch (amd64/arm64/riscv64)
- Formats supported distribution (native/wasm/cuda/npu/fpga/asic)
- Accelerator inventory (GPU/NPU/FPGA/ASIC counts & top models)
- Network fabric mix and link characteristics (where present)
- CPU/memory/storage aggregates (min/mean/max)
- Health hints (crashes, thermal derate, SSD wear)
- Top-N leaderboards (CPU capacity, VRAM, accelerator score)

Outputs
-------
- Console summary (always)
- CSV inventory (optional): one row per node with key fields
- JSON summary (optional): aggregates + leaderboards
- Markdown report (optional): human-readable overview

Assumptions
-----------
Matches fields from the provided node schema and the gen_nodes.py generator.
Missing fields are handled gracefully.
"""

from __future__ import annotations
import argparse, json, math, statistics
from pathlib import Path
from typing import Any, Dict, List, Tuple
import yaml
from collections import Counter, defaultdict

# ----------------- IO -----------------

def load_yaml(p: Path) -> Any:
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_nodes(nodes_dir: Path) -> List[Dict[str, Any]]:
    nodes = []
    for f in sorted(nodes_dir.glob("*.yaml")):
        try:
            nodes.append(load_yaml(f))
        except Exception as e:
            print(f"[warn] could not load {f.name}: {e}")
    return nodes

# ----------------- helpers -----------------

def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default

def cpu_capacity(node: Dict[str, Any]) -> float:
    cpu = node.get("cpu", {})
    cores = safe_float(cpu.get("cores"), 0)
    ghz   = safe_float(cpu.get("base_ghz"), 0)
    derate = safe_float(node.get("health", {}).get("thermal_derate"), 0.0)
    return max(0.0, cores * ghz * (1.0 - derate))

def accel_score(node: Dict[str, Any]) -> float:
    g = node.get("gpu") or {}
    return safe_float(g.get("accel_score"), 0.0)

def vram_gb(node: Dict[str, Any]) -> float:
    g = node.get("gpu") or {}
    return safe_float(g.get("vram_gb"), 0.0)

def node_row(node: Dict[str, Any]) -> Dict[str, Any]:
    net = node.get("network", {})
    cpu = node.get("cpu", {})
    mem = node.get("memory", {})
    st  = node.get("storage", {})
    batt= node.get("battery", {})
    acc = node.get("accelerators", {})
    dpu = node.get("dpu", {})
    cxl = node.get("cxl", {})
    gpu = node.get("gpu", {})

    return {
        "name": node.get("name"),
        "class": node.get("class"),
        "arch": node.get("arch"),
        "formats": ",".join(node.get("formats_supported") or []),
        "cpu_cores": cpu.get("cores"),
        "cpu_base_ghz": cpu.get("base_ghz"),
        "cpu_turbo_ghz": cpu.get("turbo_ghz"),
        "cpu_capacity": round(cpu_capacity(node), 3),
        "ram_gb": mem.get("ram_gb"),
        "ecc": mem.get("ecc"),
        "ddr_gen": mem.get("ddr_gen"),
        "storage_type": st.get("type"),
        "storage_size_gb": st.get("size_gb"),
        "tbw_pct_used": st.get("tbw_pct_used"),
        "gpu_vendor": gpu.get("vendor"),
        "gpu_model": gpu.get("model"),
        "gpu_vram_gb": gpu.get("vram_gb"),
        "gpu_accel_score": gpu.get("accel_score"),
        "npu": acc.get("npu"),
        "npu_tops": acc.get("npu_tops"),
        "fpga": acc.get("fpga_card"),
        "asic": acc.get("asic_kind"),
        "dpu": dpu.get("type"),
        "cxl": cxl.get("type"),
        "fabric": net.get("fabric"),
        "nic_speed_gbps": net.get("speed_gbps"),
        "mtu": net.get("mtu_bytes"),
        "labels_zone": (node.get("labels") or {}).get("zone"),
        "labels_trust": (node.get("labels") or {}).get("trust"),
        "down": (node.get("_sim") or {}).get("down", False),
        "thermal_derate": (node.get("health") or {}).get("thermal_derate"),
        "ram_errors_million_hours": (node.get("health") or {}).get("ram_errors_million_hours"),
        "last_week_crashes": (node.get("health") or {}).get("last_week_crashes"),
        "battery_present": batt.get("present"),
        "battery_health_pct": batt.get("health_pct"),
    }

def agg_stats(values: List[float]) -> Dict[str, float]:
    vals = [v for v in values if isinstance(v, (int, float)) and not math.isnan(v)]
    if not vals:
        return {"min": 0, "mean": 0, "p50": 0, "p90": 0, "max": 0}
    vals_sorted = sorted(vals)
    def p(q: float) -> float:
        idx = max(0, min(len(vals_sorted)-1, int(round(q*(len(vals_sorted)-1)))))
        return float(vals_sorted[idx])
    return {
        "min": float(vals_sorted[0]),
        "mean": float(statistics.fmean(vals)),
        "p50": p(0.5),
        "p90": p(0.9),
        "max": float(vals_sorted[-1]),
    }

# ----------------- summarizer -----------------

def summarize(nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_class = Counter(n.get("class") for n in nodes)
    by_arch  = Counter(n.get("arch") for n in nodes)
    formats_flat = []
    fabrics = Counter()
    gpu_models = Counter()
    npu_kinds = Counter()
    fpga_cards = Counter()
    asic_kinds = Counter()

    cpu_caps = []
    ram_gbs  = []
    vram_gbs = []
    trust_vals = []

    worn_out = 0
    crashy   = 0

    for n in nodes:
        # formats
        fmts = n.get("formats_supported") or []
        formats_flat.extend(fmts)
        # net fabrics
        fabrics[n.get("network", {}).get("fabric", "unknown")] += 1
        # accelerators
        g = n.get("gpu") or {}
        if (g.get("type") != "none") and g.get("model"):
            gpu_models[g.get("model")] += 1
        acc = n.get("accelerators") or {}
        npu = acc.get("npu", "none")
        if npu and npu != "none":
            npu_kinds[npu] += 1
        fpga = acc.get("fpga_card", "none")
        if fpga and fpga != "none":
            fpga_cards[fpga] += 1
        asic = acc.get("asic_kind", "none")
        if asic and asic != "none":
            asic_kinds[asic] += 1

        # aggregates
        cpu_caps.append(cpu_capacity(n))
        ram_gbs.append(safe_float((n.get("memory") or {}).get("ram_gb")))
        vram_gbs.append(vram_gb(n))
        t = (n.get("labels") or {}).get("trust")
        try:
            trust_vals.append(float(t))
        except Exception:
            pass

        # health
        st = n.get("storage") or {}
        if safe_float(st.get("tbw_pct_used")) > 90:
            worn_out += 1
        if (n.get("health") or {}).get("last_week_crashes", 0) not in (0, None):
            crashy += 1

    fmt_counts = Counter(formats_flat)
    summary = {
        "total_nodes": len(nodes),
        "by_class": dict(by_class),
        "by_arch": dict(by_arch),
        "formats_supported": dict(fmt_counts),
        "fabrics": dict(fabrics),
        "accelerators": {
            "gpu_models": dict(gpu_models.most_common(10)),
            "npu_kinds": dict(npu_kinds),
            "fpga_cards": dict(fpga_cards),
            "asic_kinds": dict(asic_kinds),
        },
        "stats": {
            "cpu_capacity": agg_stats(cpu_caps),
            "ram_gb": agg_stats(ram_gbs),
            "gpu_vram_gb": agg_stats(vram_gbs),
            "trust": agg_stats(trust_vals) if trust_vals else None,
        },
        "health": {
            "worn_out_storage_count": worn_out,
            "crashy_nodes_count": crashy,
        },
        "leaders": {
            "top_cpu_capacity": top_n(nodes, key=lambda n: cpu_capacity(n), n=8,
                                      fields=["name","class","arch","cpu.cores","cpu.base_ghz","cpu.turbo_ghz"]),
            "top_vram": top_n(nodes, key=lambda n: vram_gb(n), n=8, fields=["name","class","gpu.vendor","gpu.model","gpu.vram_gb"]),
            "top_accel_score": top_n(nodes, key=lambda n: accel_score(n), n=8, fields=["name","class","gpu.vendor","gpu.model","gpu.accel_score"]),
        }
    }
    return summary

def _get_field(d: Dict[str, Any], dotted: str) -> Any:
    cur = d
    for p in dotted.split("."):
        if isinstance(cur, dict):
            cur = cur.get(p)
        else:
            return None
    return cur

def top_n(nodes: List[Dict[str, Any]], key, n: int, fields: List[str]) -> List[Dict[str, Any]]:
    arr = sorted(nodes, key=key, reverse=True)[:n]
    out: List[Dict[str, Any]] = []
    for x in arr:
        row = {}
        for f in fields:
            if "." in f:
                row[f] = _get_field(x, f)
            else:
                row[f] = x.get(f)
        out.append(row)
    return out

# ----------------- exports -----------------

def export_csv(path: Path, nodes: List[Dict[str, Any]]):
    import csv
    rows = [node_row(n) for n in nodes]
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

def export_json(path: Path, summary: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2))

def export_md(path: Path, summary: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    s = summary
    def line_dict(d: Dict[str, Any]) -> str:
        return " | ".join([f"**{k}**: {v}" for k,v in d.items()])

    md = []
    md.append(f"# Nodes Summary\n")
    md.append(f"- Total nodes: **{s['total_nodes']}**\n")
    md.append(f"## By Class\n")
    md.append(", ".join([f"**{k}**: {v}" for k,v in s["by_class"].items()]) + "\n")
    md.append(f"## By Arch\n")
    md.append(", ".join([f"**{k}**: {v}" for k,v in s["by_arch"].items()]) + "\n")
    md.append(f"## Formats Supported\n")
    md.append(", ".join([f"**{k}**: {v}" for k,v in s["formats_supported"].items()]) + "\n")

    md.append("## Network Fabrics\n")
    md.append(", ".join([f"**{k}**: {v}" for k,v in s["fabrics"].items()]) + "\n")

    md.append("## Accelerators (Top GPU models)\n")
    if s["accelerators"]["gpu_models"]:
        md.append(", ".join([f"**{k}**: {v}" for k,v in s["accelerators"]["gpu_models"].items()]) + "\n")
    else:
        md.append("_No GPUs detected_\n")

    md.append("## Stats\n")
    for k, v in s["stats"].items():
        if v is None: 
            continue
        md.append(f"- **{k}**: min={v['min']:.2f}, mean={v['mean']:.2f}, p50={v['p50']:.2f}, p90={v['p90']:.2f}, max={v['max']:.2f}")

    md.append("\n## Health\n")
    md.append(f"- Worn-out storage (>90% TBW): **{s['health']['worn_out_storage_count']}**")
    md.append(f"- Crashy nodes (last week): **{s['health']['crashy_nodes_count']}**")

    def table(items: List[Dict[str, Any]]):
        if not items: return "_none_"
        headers = list(items[0].keys())
        out = ["\n|" + "|".join(headers) + "|", "|" + "|".join(["---"]*len(headers)) + "|"]
        for r in items:
            out.append("|" + "|".join([str(r.get(h, "")) for h in headers]) + "|")
        return "\n".join(out)

    md.append("\n## Leaders: Top CPU Capacity\n")
    md.append(table(s["leaders"]["top_cpu_capacity"]))

    md.append("\n## Leaders: Top VRAM\n")
    md.append(table(s["leaders"]["top_vram"]))

    md.append("\n## Leaders: Top Accelerator Score\n")
    md.append(table(s["leaders"]["top_accel_score"]))
    md.append("\n")

    path.write_text("\n".join(md), encoding="utf-8")

# ----------------- CLI -----------------

def main():
    ap = argparse.ArgumentParser(description="Summarize node descriptors")
    ap.add_argument("--dir", default="nodes", help="Directory with per-node YAMLs")
    ap.add_argument("--csv", default=None, help="Export inventory CSV path")
    ap.add_argument("--json", default=None, help="Export summary JSON path")
    ap.add_argument("--md", default=None, help="Export Markdown report path")
    args = ap.parse_args()

    nodes_dir = Path(args.dir)
    nodes = load_nodes(nodes_dir)
    if not nodes:
        print(f"[error] no nodes found in {nodes_dir.as_posix()}")
        return 2

    summary = summarize(nodes)

    # Console summary
    print(f"Total nodes: {summary['total_nodes']}")
    print("By class:", summary["by_class"])
    print("By arch:", summary["by_arch"])
    print("Formats:", summary["formats_supported"])
    print("Fabrics:", summary["fabrics"])
    print("Top GPU models:", summary["accelerators"]["gpu_models"])
    print("CPU capacity (min/mean/p90/max): "
          f"{summary['stats']['cpu_capacity']['min']:.2f}/"
          f"{summary['stats']['cpu_capacity']['mean']:.2f}/"
          f"{summary['stats']['cpu_capacity']['p90']:.2f}/"
          f"{summary['stats']['cpu_capacity']['max']:.2f}")
    print("VRAM GB (min/mean/p90/max): "
          f"{summary['stats']['gpu_vram_gb']['min']:.2f}/"
          f"{summary['stats']['gpu_vram_gb']['mean']:.2f}/"
          f"{summary['stats']['gpu_vram_gb']['p90']:.2f}/"
          f"{summary['stats']['gpu_vram_gb']['max']:.2f}")
    if summary['stats'].get('trust'):
        t = summary['stats']['trust']
        print("Trust (min/mean/p90/max): "
              f"{t['min']:.2f}/{t['mean']:.2f}/{t['p90']:.2f}/{t['max']:.2f}")
    print("Health: worn_out_storage:", summary["health"]["worn_out_storage_count"],
          "crashy_nodes:", summary["health"]["crashy_nodes_count"])

    # Exports
    if args.csv:
        export_csv(Path(args.csv), nodes)
        print("CSV:", Path(args.csv).as_posix())
    if args.json:
        export_json(Path(args.json), summary)
        print("JSON:", Path(args.json).as_posix())
    if args.md:
        export_md(Path(args.md), summary)
        print("Markdown:", Path(args.md).as_posix())

    return 0

if __name__ == "__main__":
    raise SystemExit(main())

