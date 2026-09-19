#!/usr/bin/env python3
"""
Turn ablation runs into the numbers the paper reports, for any set of domains.

Reads the manifests written by analysis/run_ablation.py and prints:

  * Table IV(a)/(b), PCMDE rows  - mean, std, min, max, range, CV per domain
  * Table XIII(a)                - mean, std, CV, pass rate per ablation variant
  * Table XIII(b)                - Pearson r and verdict agreement vs the full pipeline

Each block is printed twice: as a readable table, and as LaTeX rows to paste
into the paper. Std is the sample standard deviation (n-1), CV = 100*std/mean,
matching the paper's reported values.

Usage:
    python analysis/ablation_analysis.py --config configs/aircraft.yaml configs/car.yaml configs/animal.yaml
    python analysis/ablation_analysis.py --dir results/car/ablation
    python analysis/ablation_analysis.py --config configs/car.yaml --csv summary.csv

Replaces the domain-hardcoded ablation_analysis_aircraft.py / _car.py, which
read the 2026-03 legacy output schema (see results/<domain>/ablation/legacy_run_2026-03/).
"""

import argparse
import csv
import json
import statistics as st
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

VARIANT_LABELS = {
    "full_pipeline": "Full Pipe.",
    "rules_only": "Rules Only",
    "llm_only": "LLM Only",
    "vlm_only": "VLM-only Det.",
    "detector_only": "Detector-only Det.",
}
VARIANT_ORDER = ["full_pipeline", "rules_only", "llm_only", "vlm_only", "detector_only"]


def pearson(xs, ys):
    """Pearson correlation without a numpy/scipy dependency."""
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0 or syy <= 0:
        return float("nan")
    return sxy / (sxx * syy) ** 0.5


def describe(scores, verdicts):
    n = len(scores)
    mean = sum(scores) / n
    std = st.stdev(scores) if n > 1 else 0.0
    passed = sum(1 for v in verdicts if v == "PASS")
    return {
        "n": n, "mean": mean, "std": std,
        "cv": (100.0 * std / mean) if mean else float("nan"),
        "min": min(scores), "max": max(scores), "range": max(scores) - min(scores),
        "pass_rate": 100.0 * passed / n,
    }


def load_domain(manifest_path: Path):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    variants = {}
    for variant, path in manifest["variants"].items():
        resolved = Path(path)
        if not resolved.is_absolute():
            resolved = (PROJECT_ROOT / path).resolve()
        if not resolved.exists():
            resolved = manifest_path.parent / Path(path).name
        package = json.loads(resolved.read_text(encoding="utf-8"))
        results = package["results"]
        variants[variant] = {
            "scores": {r["image_id"]: r["overall_score"] for r in results},
            "verdicts": {r["image_id"]: r["verdict"] for r in results},
            "llm_failures": package["statistics"].get("llm_failures", 0),
        }
    return manifest, variants


def find_manifests(args) -> list:
    paths = []
    for config_path in args.config or []:
        from core.config_manager import ConfigManager
        cfg = ConfigManager.load(config_path)
        rules = json.loads(Path(cfg.require("paths.rules_file")).read_text(encoding="utf-8"))
        domain = rules.get("object_type", "domain")
        directory = Path(cfg.get("paths.ablation_dir") or f"./results/{domain}/ablation")
        paths.append(directory / f"{domain}_ablation_manifest.json")
    for directory in args.dir or []:
        paths.extend(sorted(Path(directory).glob("*_ablation_manifest.json")))
    return [p for p in paths if p.exists()] or paths


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", nargs="+", help="Domain config YAML(s)")
    ap.add_argument("--dir", nargs="+", help="Ablation directory/directories")
    ap.add_argument("--csv", help="Also write every statistic to this CSV")
    args = ap.parse_args(argv)
    if not args.config and not args.dir:
        ap.error("provide --config and/or --dir")

    manifests = find_manifests(args)
    missing = [p for p in manifests if not p.exists()]
    if missing:
        print("Missing ablation manifest(s):")
        for p in missing:
            print(f"  {p}   (run: python analysis/run_ablation.py --config <config>)")
        manifests = [p for p in manifests if p.exists()]
    if not manifests:
        return 1

    domains, rows = [], []
    for manifest_path in manifests:
        manifest, variants = load_domain(manifest_path)
        domains.append((manifest, variants))

    # ---------------- Table IV: PCMDE score distribution per domain ----------
    print("\n" + "=" * 78)
    print("TABLE IV - PCMDE rows (full pipeline)")
    print("=" * 78)
    print(f"{'Domain':<12}{'N':>5}{'Mean':>8}{'Std':>8}{'Min':>8}{'Max':>8}{'Range':>8}{'CV%':>8}")
    latex_iv = []
    for manifest, variants in domains:
        full = variants.get("full_pipeline")
        if not full:
            continue
        ids = sorted(full["scores"])
        stats = describe([full["scores"][i] for i in ids], [full["verdicts"][i] for i in ids])
        domain = manifest["domain"]
        print(f"{domain:<12}{stats['n']:>5}{stats['mean']:>8.1f}{stats['std']:>8.2f}"
              f"{stats['min']:>8.1f}{stats['max']:>8.1f}{stats['range']:>8.1f}{stats['cv']:>8.2f}")
        latex_iv.append(f"{domain.capitalize()} & PCMDE & {stats['mean']:.1f} & {stats['std']:.2f} & "
                        f"{stats['min']:.1f} & {stats['max']:.1f} \\\\   % (b): range {stats['range']:.1f}, CV {stats['cv']:.1f}")
        rows.append({"table": "IV", "domain": domain, "variant": "full_pipeline", **stats})
    print("\nLaTeX rows (Table IV, PCMDE only - baseline metric rows come from the "
          "cross-check notebooks):")
    for line in latex_iv:
        print("  " + line)

    # ---------------- Table XIII(a): ablation statistics ---------------------
    print("\n" + "=" * 78)
    print("TABLE XIII(a) - ablation statistics")
    print("=" * 78)
    print(f"{'Domain':<12}{'Variant':<20}{'N':>5}{'Mean':>8}{'Std':>8}{'CV%':>8}{'Pass%':>8}")
    latex_xiii_a = []
    for manifest, variants in domains:
        for variant in VARIANT_ORDER:
            data = variants.get(variant)
            if not data:
                continue
            ids = sorted(data["scores"])
            stats = describe([data["scores"][i] for i in ids], [data["verdicts"][i] for i in ids])
            print(f"{manifest['domain']:<12}{VARIANT_LABELS[variant]:<20}{stats['n']:>5}"
                  f"{stats['mean']:>8.1f}{stats['std']:>8.2f}{stats['cv']:>8.1f}{stats['pass_rate']:>8.1f}")
            latex_xiii_a.append(
                f"{manifest['domain'].capitalize()} & {VARIANT_LABELS[variant]} & {stats['mean']:.1f} & "
                f"{stats['std']:.2f} & {stats['cv']:.1f} & {stats['pass_rate']:.1f}\\% \\\\")
            rows.append({"table": "XIII(a)", "domain": manifest["domain"], "variant": variant, **stats})
    print("\nLaTeX rows (Table XIII(a)):")
    for line in latex_xiii_a:
        print("  " + line)

    # ---------------- Table XIII(b): agreement with the full pipeline -------
    print("\n" + "=" * 78)
    print("TABLE XIII(b) - agreement with the full pipeline")
    print("=" * 78)
    print(f"{'Domain':<12}{'Variant':<20}{'Pearson r':>11}{'Verdict agree %':>17}")
    latex_xiii_b = []
    for manifest, variants in domains:
        full = variants.get("full_pipeline")
        if not full:
            continue
        for variant in VARIANT_ORDER[1:]:
            data = variants.get(variant)
            if not data:
                continue
            ids = sorted(set(full["scores"]) & set(data["scores"]))
            r = pearson([full["scores"][i] for i in ids], [data["scores"][i] for i in ids])
            agree = 100.0 * sum(1 for i in ids if full["verdicts"][i] == data["verdicts"][i]) / len(ids)
            print(f"{manifest['domain']:<12}{VARIANT_LABELS[variant]:<20}{r:>11.3f}{agree:>17.1f}")
            latex_xiii_b.append(
                f"{VARIANT_LABELS[variant]} & {manifest['domain']} & {r:.3f} & {agree:.1f} \\\\")
            rows.append({"table": "XIII(b)", "domain": manifest["domain"], "variant": variant,
                         "pearson_r": r, "verdict_agreement": agree, "n": len(ids)})
    print("\nLaTeX rows (Table XIII(b), one line per domain/variant):")
    for line in latex_xiii_b:
        print("  " + line)

    # ---------------- health warnings ---------------------------------------
    problems = []
    for manifest, variants in domains:
        if not manifest.get("llm_enabled", True):
            problems.append(f"{manifest['domain']}: ran with the LLM stage DISABLED - "
                            "these numbers are not reportable.")
        for variant, data in variants.items():
            if data["llm_failures"]:
                problems.append(f"{manifest['domain']}/{variant}: {data['llm_failures']} image(s) "
                                "had no valid LLM reply.")
        counts = {v: len(d["scores"]) for v, d in variants.items()}
        if len(set(counts.values())) > 1:
            problems.append(f"{manifest['domain']}: variants cover different image counts {counts}.")
    if problems:
        print("\n" + "!" * 78)
        print("CHECK BEFORE REPORTING")
        print("!" * 78)
        for problem in problems:
            print("  - " + problem)

    if args.csv:
        fieldnames = sorted({k for row in rows for k in row})
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved -> {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
