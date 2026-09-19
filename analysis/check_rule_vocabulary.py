#!/usr/bin/env python3
"""
Check that a domain's rules JSON only references components the detector
actually emits.

Every rule referencing a component absent from the ensemble's fused_counts
becomes an unevaluable violation on every image. If such a rule has
severity="critical", the critical cap fires for the whole dataset and every
image collapses to the same score.

Run this before evaluating a new domain, or after changing a detector's
class list.

Usage:
    python analysis/check_rule_vocabulary.py --config configs/aircraft.yaml
    python analysis/check_rule_vocabulary.py \
        --rules domains/car/physical_rules_car.json \
        --ensemble data/car/ensemble/ensemble_car.json

Exit code 0 = clean, 1 = at least one referenced component is missing.
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def referenced_components(rules_json: dict) -> set:
    """Every component name any rule mentions, including critical_presence lists."""
    names = set()
    for rule in rules_json.get("rules", []):
        for key in ("subject_component", "reference_component"):
            if rule.get(key):
                names.add(rule[key])
        params = rule.get("parameters") or {}
        for comp in params.get("required_components", []) or []:
            names.add(comp)
    return names


def emitted_components(ensemble_json: dict) -> set:
    """Every component key appearing in any image's fused_counts."""
    names = set()
    for result in ensemble_json.get("results", []):
        counts = result.get("ensemble_output", {}).get("fused_counts", {})
        names |= set(counts.keys())
    return names


def critical_components(rules_json: dict) -> set:
    """Components referenced by critical-severity rules (these cap the score)."""
    names = set()
    for rule in rules_json.get("rules", []):
        if rule.get("severity") != "critical":
            continue
        for key in ("subject_component", "reference_component"):
            if rule.get(key):
                names.add(rule[key])
        params = rule.get("parameters") or {}
        for comp in params.get("required_components", []) or []:
            names.add(comp)
    return names


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="Domain config YAML; reads rules_file and ensemble_json from it")
    ap.add_argument("--rules", help="Path to the rules JSON (alternative to --config)")
    ap.add_argument("--ensemble", help="Path to the ensemble JSON (alternative to --config)")
    args = ap.parse_args()

    if args.config:
        from core.config_manager import ConfigManager
        cfg = ConfigManager.load(args.config)
        rules_path = cfg.require("paths.rules_file")
        ensemble_path = cfg.require("paths.ensemble_json")
    elif args.rules and args.ensemble:
        rules_path, ensemble_path = args.rules, args.ensemble
    else:
        ap.error("provide --config, or both --rules and --ensemble")

    rules_json = json.loads(Path(rules_path).read_text(encoding="utf-8"))
    ensemble_json = json.loads(Path(ensemble_path).read_text(encoding="utf-8"))

    referenced = referenced_components(rules_json)
    emitted = emitted_components(ensemble_json)
    critical = critical_components(rules_json)

    missing = sorted(referenced - emitted)
    unused = sorted(emitted - referenced)
    missing_critical = sorted(set(missing) & critical)

    print(f"rules    : {rules_path}")
    print(f"ensemble : {ensemble_path}")
    print(f"  rules reference : {len(referenced)} components")
    print(f"  ensemble emits  : {len(emitted)} components")
    print()

    if missing:
        print(f"  MISSING ({len(missing)}) - referenced by rules, never emitted by the detector:")
        for name in missing:
            flag = "  <-- CRITICAL, caps every image" if name in missing_critical else ""
            print(f"      {name}{flag}")
    else:
        print("  MISSING: none")

    print()
    if unused:
        print(f"  unused ({len(unused)}) - emitted by the detector, no rule uses them:")
        print(f"      {', '.join(unused)}")
    else:
        print("  unused: none")

    # Case-collision detection: components differing only by case/plural.
    print()
    collisions = {}
    for name in emitted:
        collisions.setdefault(name.lower().rstrip("s"), []).append(name)
    dupes = {k: v for k, v in collisions.items() if len(v) > 1}
    if dupes:
        print("  CASE/PLURAL COLLISIONS - fused as separate components, counts get split:")
        for _, variants in sorted(dupes.items()):
            print(f"      {sorted(variants)}")
    else:
        print("  case/plural collisions: none")

    print()
    if missing_critical:
        print("  RESULT: BROKEN - a critical rule references a component that is never")
        print("          detected, so the critical cap fires on every image and all")
        print("          scores collapse to critical_violation_cap.")
    elif missing:
        print("  RESULT: DEGRADED - some rules are unevaluable on every image.")
    else:
        print("  RESULT: OK")

    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
