#!/usr/bin/env python3
"""
Table IV rows: dispersion statistics for PCMDE and every baseline metric.

Reads whatever exists in results/<domain>/crosscheck/ plus the PCMDE scores,
and prints mean / std / min / max / range / CV per metric per domain, as text
and as LaTeX rows. Missing metrics are listed as missing rather than skipped
silently, so it is obvious what still has to be run.

    python metrics/crosscheck_summary.py --config configs/aircraft.yaml configs/animal.yaml configs/car.yaml
    python metrics/crosscheck_summary.py --config configs/car.yaml --vqascore path/to/vqascore_car.json

Std is the sample standard deviation (n-1) and CV = 100*std/mean, matching
analysis/ablation_analysis.py and the paper.
"""

import argparse
import csv
import json
import statistics as st
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.config_manager import ConfigManager                      # noqa: E402

# Metrics are reported on their native scale; the paper puts them all on 0-100.
SCALES = {"PCMDE": 1.0, "CLIPScore": 1.0, "VQAScore": 100.0,
          "SigLIP-2": 100.0, "FLEUR": 100.0}


def describe(name, scores):
    n = len(scores)
    mean = sum(scores) / n
    std = st.stdev(scores) if n > 1 else 0.0
    return {"metric": name, "n": n, "mean": mean, "std": std,
            "min": min(scores), "max": max(scores), "range": max(scores) - min(scores),
            "cv": (100.0 * std / mean) if mean else float("nan")}


def from_csv(path, column):
    if not path.exists():
        return None
    with open(path, newline="", encoding="utf-8") as fh:
        vals = [float(r[column]) for r in csv.DictReader(fh)
                if r.get(column) not in (None, "", "None")]
    return vals or None


def from_json_scores(path, keys):
    """Pull per-image scores out of a metric JSON, tolerating several shapes."""
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data
    if isinstance(data, dict):
        for key in ("detailed_results", "results", "scores", "per_image"):
            if isinstance(data.get(key), list):
                rows = data[key]
                break
        else:
            rows = [{"score": v} for v in data.values() if isinstance(v, (int, float))]
    vals = []
    for row in rows:
        if isinstance(row, (int, float)):
            vals.append(float(row))
        elif isinstance(row, dict):
            for key in keys:
                if isinstance(row.get(key), (int, float)):
                    vals.append(float(row[key]))
                    break
    return vals or None


def collect(domain, crosscheck_dir, scores_json, vqascore_path):
    found, missing = [], []
    sources = [
        ("PCMDE", lambda: from_json_scores(scores_json, ("overall_score",))),
        ("CLIPScore", lambda: from_csv(crosscheck_dir / f"clipscore_results_{domain}.csv", "CLIPScore")),
        ("VQAScore", lambda: from_json_scores(
            vqascore_path or crosscheck_dir / f"vqascore_{domain}.json",
            ("vqascore", "vqa_score", "score"))),
        ("SigLIP-2", lambda: from_json_scores(crosscheck_dir / f"siglip2_scores_{domain}.json",
                                              ("siglip_score",))),
        ("FLEUR", lambda: from_csv(crosscheck_dir / f"fleur_scores_{domain}.csv", "fleur_score")),
    ]
    for name, loader in sources:
        try:
            values = loader()
        except Exception as exc:                                   # noqa: BLE001
            print(f"  ! {domain} {name}: {exc}")
            values = None
        if values:
            found.append(describe(name, [v * SCALES[name] for v in values]))
        else:
            missing.append(name)
    return found, missing


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", nargs="+", required=True, help="Domain config YAML(s)")
    ap.add_argument("--vqascore", nargs="*", default=[],
                    help="VQAScore JSON(s) if they live outside results/<domain>/crosscheck/")
    ap.add_argument("--csv", help="Also write every statistic to this CSV")
    args = ap.parse_args()

    vqa_paths = [Path(p) for p in args.vqascore]
    rows = []
    for config_path in args.config:
        cfg = ConfigManager.load(config_path)
        rules = json.loads(Path(cfg.require("paths.rules_file")).read_text(encoding="utf-8"))
        domain = rules.get("object_type", "domain")
        scores_json = Path(cfg.get("paths.output_json") or
                           f"results/{domain}/scores/{domain}_pcmde_scores.json")
        vqa = next((p for p in vqa_paths if domain in p.name), None)
        found, missing = collect(domain, Path(f"results/{domain}/crosscheck"), scores_json, vqa)

        print(f"\n===== {domain} " + "=" * 52)
        print(f"{'Metric':<12}{'n':>5}{'Mean':>9}{'Std':>8}{'Min':>8}{'Max':>8}{'Range':>8}{'CV%':>8}")
        for s in found:
            print(f"{s['metric']:<12}{s['n']:>5}{s['mean']:>9.2f}{s['std']:>8.2f}"
                  f"{s['min']:>8.2f}{s['max']:>8.2f}{s['range']:>8.2f}{s['cv']:>8.2f}")
            rows.append({"domain": domain, **s})
        if missing:
            print(f"  not yet run: {', '.join(missing)}")

    print("\n===== LaTeX rows (metric & mean & std & min & max & range & CV) =====")
    for r in rows:
        print(f"{r['domain']} & {r['metric']} & {r['mean']:.2f} & {r['std']:.2f} & "
              f"{r['min']:.2f} & {r['max']:.2f} & {r['range']:.2f} & {r['cv']:.2f} \\\\")

    if args.csv and rows:
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote {args.csv}")


if __name__ == "__main__":
    main()
