#!/usr/bin/env python3
"""
Export & summarize Monte-Carlo CSVs.

Inputs
------
One or more CSVs produced by sim/montecarlo.py. Minimum expected columns:
  - trial (int)
  - latency_ms (float)
  - slo_ok (0/1)
Optional columns (auto-detected if present), e.g.:
  - scenario, seed, planner, dt_mode, etc.

Outputs (written to --outdir)
-----------------------------
- summary.csv
    Per input (run_label), overall mean / pXX / SLO hit-rate.
- grouped.csv   (if --group-by provided and columns exist)
    Same metrics broken down by group-by keys.
- cdf.csv
    Empirical CDF points for latency per run_label (and group keys if grouped).
- merged.csv    (optional) Concatenated raw rows with run_label attached.

Usage
-----
# Basic (single input)
python3 tools/export_csv.py --inputs sim/montecarlo_results.csv --outdir sim/exports

# Multiple inputs with labels
python3 tools/export_csv.py --inputs sim/mc_a.csv sim/mc_b.csv --labels A B --outdir sim/exports

# Group by scenario column and compute custom percentiles
python3 tools/export_csv.py --inputs sim/mc.csv --group-by scenario --percentiles 50,90,95,99 --outdir sim/exports

# Also write merged raw csv
python3 tools/export_csv.py --inputs sim/mc1.csv sim/mc2.csv --labels base chaos --write-merged --outdir sim/exports
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd

# ----------------------- helpers -----------------------

def percentile_series(s: pd.Series, qs: List[float]) -> Dict[str, float]:
    q = np.array(qs) / 100.0
    vals = s.quantile(q, interpolation="nearest")
    return {f"p{int(qs[i])}": float(vals.iloc[i]) for i in range(len(qs))}

def summarize_frame(df: pd.DataFrame, percentiles: List[float]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["count","mean","p50","p90","p95","p99","slo_hit_rate"])
    out: Dict[str, Any] = {}
    out["count"] = int(df.shape[0])
    out["mean"]  = float(df["latency_ms"].mean())
    out.update(percentile_series(df["latency_ms"], percentiles))
    slo = df.get("slo_ok")
    if slo is not None:
        out["slo_hit_rate"] = float(slo.mean())
    else:
        out["slo_hit_rate"] = np.nan
    return pd.DataFrame([out])

def empirical_cdf(series: pd.Series, points: int = 200) -> pd.DataFrame:
    s = series.dropna().sort_values().to_numpy()
    if s.size == 0:
        return pd.DataFrame({"latency_ms": [], "cdf": []})
    xs = np.linspace(0, 1, num=points, endpoint=True)
    idx = np.clip((xs * (s.size - 1)).astype(int), 0, s.size - 1)
    vals = s[idx]
    return pd.DataFrame({"latency_ms": vals, "cdf": xs})

def safe_read_csv(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        # ensure expected cols
        if "latency_ms" not in df.columns:
            raise ValueError("missing 'latency_ms' column")
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to read {path}: {e}")

# ----------------------- main -----------------------

def main():
    ap = argparse.ArgumentParser(description="Export & summarize Monte-Carlo CSVs")
    ap.add_argument("--inputs", nargs="+", required=True, help="One or more montecarlo_results CSVs")
    ap.add_argument("--labels", nargs="*", default=None, help="Optional labels, one per input")
    ap.add_argument("--group-by", default=None, help="Comma-separated columns to group by (must exist in CSV)")
    ap.add_argument("--percentiles", default="50,90,95,99", help="Comma-separated percentile list")
    ap.add_argument("--outdir", default="sim/exports", help="Output directory")
    ap.add_argument("--write-merged", action="store_true", help="Also write merged.csv with run_label")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    inputs = [Path(p) for p in args.inputs]
    labels = args.labels
    if labels and len(labels) != len(inputs):
        raise SystemExit("Error: number of --labels must match number of --inputs")

    pct = [float(x.strip()) for x in args.percentiles.split(",") if x.strip()]
    for x in pct:
        if x <= 0 or x >= 100:
            raise SystemExit("Error: percentiles must be between 0 and 100 (exclusive)")

    # Read & tag
    frames: List[pd.DataFrame] = []
    for i, p in enumerate(inputs):
        df = safe_read_csv(p).copy()
        run_label = labels[i] if labels else p.stem
        df["run_label"] = run_label
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True, sort=False)

    # Overall summaries per run_label
    rows = []
    cdf_blocks = []

    for label, sub in merged.groupby("run_label"):
        summ = summarize_frame(sub, pct)
        summ.insert(0, "run_label", label)
        rows.append(summ)

        cdf = empirical_cdf(sub["latency_ms"])
        cdf.insert(0, "run_label", label)
        cdf_blocks.append(cdf)

    summary_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    cdf_df = pd.concat(cdf_blocks, ignore_index=True) if cdf_blocks else pd.DataFrame()

    # Grouped summaries (optional)
    grouped_df = pd.DataFrame()
    if args.group_by:
        groups = [g.strip() for g in args.group_by.split(",") if g.strip()]
        missing = [g for g in groups if g not in merged.columns]
        if missing:
            print(f"[warn] group-by columns not found: {missing} — skipping grouped summary")
        else:
            gb_rows = []
            gb_cdfs = []
            for (label, *keys), sub in merged.groupby(["run_label", *groups], dropna=False):
                summ = summarize_frame(sub, pct)
                summ.insert(0, "run_label", label)
                for gi, gname in enumerate(groups):
                    summ.insert(gi+1, gname, keys[gi])
                gb_rows.append(summ)

                cdf = empirical_cdf(sub["latency_ms"])
                cdf.insert(0, "run_label", label)
                for gi, gname in enumerate(groups):
                    cdf.insert(gi+1, gname, keys[gi])
                gb_cdfs.append(cdf)

            if gb_rows:
                grouped_df = pd.concat(gb_rows, ignore_index=True)
            if gb_cdfs:
                cdf_df = pd.concat([cdf_df] + gb_cdfs, ignore_index=True) if not cdf_df.empty else pd.concat(gb_cdfs, ignore_index=True)

    # Write outputs
    summary_path = outdir / "summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print("Wrote:", summary_path.as_posix())

    if not cdf_df.empty:
        cdf_path = outdir / "cdf.csv"
        cdf_df.to_csv(cdf_path, index=False)
        print("Wrote:", cdf_path.as_posix())

    if not grouped_df.empty:
        grouped_path = outdir / "grouped.csv"
        grouped_df.to_csv(grouped_path, index=False)
        print("Wrote:", grouped_path.as_posix())

    if args.write_merged:
        merged_path = outdir / "merged.csv"
        merged.to_csv(merged_path, index=False)
        print("Wrote:", merged_path.as_posix())

if __name__ == "__main__":
    main()

