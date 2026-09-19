#!/usr/bin/env python3
"""
Produce the five ablation variants for one domain (paper Sec. VI-F).

    full_pipeline    detector + 2 VLMs fused, S = 0.6*S_rules + 0.4*S_LLM
    rules_only       same evidence, weights (rules 1.0, LLM 0.0)
    llm_only         same evidence, weights (rules 0.0, LLM 1.0)
    vlm_only         evidence from the two VLMs only (no detector, no boxes)
    detector_only    evidence from the detector only (counts + boxes)

rules_only and llm_only are re-weightings of the full run, so they are derived
from it rather than re-queried: "for the rules-only and LLM-only settings we
changed only the scoring weights in the pipeline". Their verdict uses the same
thresholds applied to the single signal that remains (PASS iff the score is at
least the pass threshold and at least the component threshold).

vlm_only and detector_only change the detection evidence, so each is a full
evaluation pass (LLM included) over its own ensemble file.

Usage:
    python analysis/run_ablation.py --config configs/car.yaml
    python analysis/run_ablation.py --config configs/animal.yaml --no-llm   # plumbing check
    python analysis/run_ablation.py --config configs/car.yaml --variants vlm_only

Outputs (into paths.ablation_dir, or --out-dir):
    ensemble_<domain>_vlm_only.json, ensemble_<domain>_detector_only.json
    <domain>_ablation_<variant>.json        one per variant
    <domain>_ablation_manifest.json         variant -> file, for the analysis script
    <domain>_per_image_scores.csv           every variant's score per image
"""

import argparse
import csv
import json
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.config_manager import ConfigManager                      # noqa: E402
from core.pipeline_factory import build_pipeline                   # noqa: E402
from pipeline.ensemble import build_ensemble, save_json            # noqa: E402

EVIDENCE_VARIANTS = ("full_pipeline", "vlm_only", "detector_only")
DERIVED_VARIANTS = ("rules_only", "llm_only")
ALL_VARIANTS = EVIDENCE_VARIANTS + DERIVED_VARIANTS


def _domain_name(config: ConfigManager) -> str:
    rules = json.loads(Path(config.require("paths.rules_file")).read_text(encoding="utf-8"))
    return rules.get("object_type", "domain")


def build_ablation_ensembles(config: ConfigManager, out_dir: Path, domain: str,
                             force: bool = False) -> dict:
    """Write the vlm_only and detector_only ensembles; return {variant: path}."""
    detections = config.get("paths.detections_json")
    vlms = list(config.get("paths.vlm_jsons") or [])
    if not detections or len(vlms) != 2:
        raise SystemExit(
            "Config needs paths.detections_json and exactly two paths.vlm_jsons "
            "to build the detection ablations (see configs/car.yaml)."
        )
    rules_path = config.require("paths.rules_file")
    paths = {}
    for variant, mode, files in (
        ("vlm_only", "vlm_only", vlms),
        ("detector_only", "detector_only", [detections]),
    ):
        target = out_dir / f"ensemble_{domain}_{variant}.json"
        paths[variant] = str(target)
        if target.exists() and not force:
            print(f"[{variant}] ensemble exists, reusing {target} (use --force to rebuild)")
            continue
        document = build_ensemble(files, mode=mode, rules_path=rules_path, verbose=False)
        save_json(document, str(target))
    return paths


def run_variant(config: ConfigManager, ensemble_path: str, output_path: Path,
                use_llm: bool, force: bool) -> dict:
    """Evaluate one evidence variant and return its result package."""
    if output_path.exists() and not force:
        print(f"[{output_path.name}] exists, reusing (use --force to re-run)")
        return json.loads(output_path.read_text(encoding="utf-8"))

    variant_config = ConfigManager(deepcopy(config.as_dict()))
    variant_config.override({"paths.ensemble_json": ensemble_path})
    if not use_llm:
        variant_config.override({"llm.enabled": False})

    runner = build_pipeline(variant_config)
    package = runner.run(
        ensemble_path=ensemble_path,
        captions_path=variant_config.get("paths.captions_csv"),
        image_col=variant_config.get("csv_mapping.image_col", "Image_path"),
        caption_col=variant_config.get("csv_mapping.caption_col", "Image_caption"),
        verbose=False,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(package, indent=2), encoding="utf-8")
    print(f"[{output_path.name}] {package['statistics']}")
    return package


def derive_single_signal(package: dict, signal: str, pass_threshold: float,
                         component_threshold: float) -> dict:
    """Re-weight a full run onto one signal ('rule' or 'llm')."""
    derived = deepcopy(package)
    key = "rule_score" if signal == "rule" else "llm_score"
    weights = ({"rule_weight": 1.0, "llm_weight": 0.0} if signal == "rule"
               else {"rule_weight": 0.0, "llm_weight": 1.0})
    for result in derived["results"]:
        score = result[key]
        result["overall_score"] = score
        result["verdict"] = ("PASS" if score >= pass_threshold
                             and score >= component_threshold else "FAIL")
        result["weights"] = dict(weights)
    n = len(derived["results"]) or 1
    passed = sum(1 for r in derived["results"] if r["verdict"] == "PASS")
    scores = [r["overall_score"] for r in derived["results"]]
    derived["statistics"].update({
        "passed": passed,
        "failed": len(scores) - passed,
        "pass_rate": round(passed / n * 100.0, 2),
        "mean_overall": round(sum(scores) / n, 2),
        "min_overall": round(min(scores), 2) if scores else 0.0,
        "max_overall": round(max(scores), 2) if scores else 0.0,
    })
    derived["metadata"]["derived_from"] = "full_pipeline"
    derived["metadata"]["ablation_weights"] = weights
    derived["metadata"]["note"] = (
        f"Re-weighting of the full run onto {key}; verdict = PASS when that score "
        f">= max(pass_threshold={pass_threshold}, component_threshold={component_threshold})."
    )
    return derived


def write_per_image_csv(packages: dict, path: Path) -> None:
    image_ids, rows = [], {}
    for variant in ALL_VARIANTS:
        package = packages.get(variant)
        if not package:
            continue
        for result in package["results"]:
            image_id = result["image_id"]
            if image_id not in rows:
                rows[image_id] = {"image_id": image_id}
                image_ids.append(image_id)
            rows[image_id][f"{variant}_score"] = round(result["overall_score"], 4)
            rows[image_id][f"{variant}_verdict"] = result["verdict"]
    columns = ["image_id"] + [f"{v}_{s}" for v in ALL_VARIANTS if v in packages
                              for s in ("score", "verdict")]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for image_id in image_ids:
            writer.writerow(rows[image_id])
    print(f"Saved -> {path}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="Domain config YAML")
    ap.add_argument("--out-dir", help="Output directory (default: paths.ablation_dir)")
    ap.add_argument("--variants", default=",".join(EVIDENCE_VARIANTS),
                    help="Evidence variants to evaluate, comma-separated. "
                         "rules_only/llm_only are always derived from full_pipeline.")
    ap.add_argument("--no-llm", action="store_true",
                    help="Run without the LLM stage. Use only to test the plumbing: "
                         "llm_only becomes meaningless and the numbers are not reportable.")
    ap.add_argument("--force", action="store_true", help="Rebuild files that already exist")
    args = ap.parse_args(argv)

    config = ConfigManager.load(args.config)
    domain = _domain_name(config)
    out_dir = Path(args.out_dir or config.get("paths.ablation_dir")
                   or f"./results/{domain}/ablation")
    out_dir.mkdir(parents=True, exist_ok=True)
    use_llm = (not args.no_llm) and bool(config.get("llm.enabled", True))
    pass_threshold = float(config.get("scoring.pass_threshold", 60.0))
    component_threshold = float(config.get("scoring.component_threshold", 40.0))
    wanted = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = set(wanted) - set(EVIDENCE_VARIANTS)
    if unknown:
        raise SystemExit(f"Unknown variant(s) {sorted(unknown)}; choose from {EVIDENCE_VARIANTS}")

    print(f"Domain      : {domain}")
    print(f"Output dir  : {out_dir}")
    print(f"LLM stage   : {'enabled' if use_llm else 'DISABLED (plumbing test only)'}")

    ensembles = {"full_pipeline": config.require("paths.ensemble_json")}
    if {"vlm_only", "detector_only"} & set(wanted):
        ensembles.update(build_ablation_ensembles(config, out_dir, domain, force=args.force))

    packages, files = {}, {}
    for variant in wanted:
        target = out_dir / f"{domain}_ablation_{variant}.json"
        packages[variant] = run_variant(config, ensembles[variant], target, use_llm, args.force)
        files[variant] = str(target)

    if "full_pipeline" in packages:
        for variant, signal in (("rules_only", "rule"), ("llm_only", "llm")):
            derived = derive_single_signal(packages["full_pipeline"], signal,
                                           pass_threshold, component_threshold)
            target = out_dir / f"{domain}_ablation_{variant}.json"
            target.write_text(json.dumps(derived, indent=2), encoding="utf-8")
            packages[variant] = derived
            files[variant] = str(target)
            print(f"[{target.name}] {derived['statistics']}")

    manifest = {
        "domain": domain,
        "created": datetime.now().isoformat(),
        "config": str(args.config),
        "llm_enabled": use_llm,
        "pass_threshold": pass_threshold,
        "component_threshold": component_threshold,
        "variants": files,
        "llm_failures": {v: p["statistics"].get("llm_failures", 0) for v, p in packages.items()},
    }
    manifest_path = out_dir / f"{domain}_ablation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved -> {manifest_path}")
    write_per_image_csv(packages, out_dir / f"{domain}_per_image_scores.csv")

    failures = sum(manifest["llm_failures"].values())
    if failures:
        print(f"\nWARNING: {failures} image evaluation(s) had no valid LLM reply. "
              "Fix the LLM setup and re-run with --force before reporting these numbers.")
    print("\nNext: python analysis/ablation_analysis.py --config " + str(args.config))
    return 0


if __name__ == "__main__":
    sys.exit(main())
