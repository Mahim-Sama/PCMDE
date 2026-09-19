"""
Batch runner.

Orchestrates end-to-end evaluation over a dataset. Responsibilities:

    1. Load ensemble results JSON
    2. Load captions CSV (optional)
    3. Validate ID matching between ensemble and captions
    4. For each image:
        a. Resolve subtype (from caption if available, else filename)
        b. Run rule evaluator
        c. Run LLM validator (if enabled)
        d. Combine scores
    5. Compute aggregate statistics
    6. Serialize results to JSON

This module is domain-agnostic. All domain-specific logic is injected via
configuration and the knowledge/vocabulary JSONs loaded upstream.
"""

import csv
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .config_manager import ConfigManager
from .evaluator import RuleEvaluator
from .final_scorer import FinalScore, FinalScorer
from .llm_validator import LLMValidator
from .rule_loader import CompiledRule
from .subtype_resolver import SubtypeResolver


# ---------------------------------------------------------------------------
# CSV loading with duplicate detection
# ---------------------------------------------------------------------------

def load_captions_csv(csv_path: str,
                      image_col: str = "Image_path",
                      caption_col: str = "Image_caption",
                      verbose: bool = True) -> Dict[str, str]:
    """Load captions from CSV, keyed by filename."""
    captions: Dict[str, str] = {}
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Captions CSV not found: {csv_path}")

    seen_rows: Dict[str, int] = {}
    with open(path, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if image_col not in reader.fieldnames:
            raise ValueError(
                f"Column '{image_col}' not in CSV. Available: {reader.fieldnames}"
            )
        if caption_col not in reader.fieldnames:
            raise ValueError(
                f"Column '{caption_col}' not in CSV. Available: {reader.fieldnames}"
            )
        for row_num, row in enumerate(reader, start=2):
            image_path = (row.get(image_col) or "").strip()
            caption = (row.get(caption_col) or "").strip()
            if not image_path or not caption:
                continue
            image_id = Path(image_path).name
            if image_id in captions:
                if verbose:
                    first = seen_rows.get(image_id, "?")
                    print(f"  Duplicate row {row_num} for '{image_id}' (first at {first}) - skipping")
                continue
            captions[image_id] = caption
            seen_rows[image_id] = row_num

    if verbose:
        print(f"  Loaded {len(captions)} unique captions from CSV")
    return captions


# ---------------------------------------------------------------------------
# Ensemble JSON loading and schema normalization
# ---------------------------------------------------------------------------

# Canonical on-disk JSON schema consumed by this framework:
#
# {
#   "fusion_metadata": {...}            (optional, top-level audit)
#   "results": [
#     {
#       "image_id": "<filename>",
#       "ensemble_output": {
#         "fused_counts":    { "<component>": <int> },       (required)
#         "yolo_detections": { "<component>": [ {bbox} ] },  (required)
#         "confidence":      <float>                          (optional)
#       },
#       "analysis": {...},              (optional, per-image audit)
#       "source_detections": {...}      (optional, per-image audit)
#     }
#   ]
# }
#
# The key "yolo_detections" is retained for backward compatibility with the
# aircraft pipeline's original output; internally we refer to the bboxes as
# detector_boxes because the detector is not required to be YOLO. The
# optional fields (fusion_metadata, analysis, source_detections) are passed
# through into the evaluation output's metadata untouched.

_ON_DISK_BOX_KEY = "yolo_detections"


def load_ensemble_json(path: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Load ensemble results and return (normalized_results, top_level_audit).

    normalized_results is a list of per-image dicts with fields:
        image_id, fused_counts, detector_boxes, ensemble_confidence,
        fusion_details (optional), agreement_analysis (optional),
        source_detections (optional).

    top_level_audit captures any keys that were at the JSON root above the
    "results" list (e.g., fusion_metadata, fusion_method, model_weights) so
    they can be surfaced in the evaluation output's metadata. Empty dict if
    the input is a bare list or has no such fields.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Ensemble JSON not found: {path}")
    with open(p, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    if isinstance(data, dict) and "results" in data:
        results = data["results"]
        top_level_audit = {k: v for k, v in data.items() if k != "results"}
    elif isinstance(data, list):
        results = data
        top_level_audit = {}
    else:
        raise ValueError(
            f"Unexpected ensemble JSON top-level structure in {path}: "
            f"expected {{'results': [...]}} or a list."
        )

    normalized: List[Dict[str, Any]] = []
    for item in results:
        ensemble_output = item.get("ensemble_output", {}) or {}

        counts = ensemble_output.get("fused_counts", {}) or {}
        boxes = ensemble_output.get(_ON_DISK_BOX_KEY, {}) or {}

        normalized.append({
            "image_id": item["image_id"],
            "fused_counts": counts,
            "detector_boxes": boxes,
            "ensemble_confidence": ensemble_output.get("confidence"),
            # Optional audit-trail fields. Pass through as-is if present,
            # None if absent. Populated into the per-image output metadata.
            "fusion_details":     (item.get("analysis") or {}).get("fusion_details"),
            "agreement_analysis": (item.get("analysis") or {}).get("agreement_analysis"),
            "confidence_breakdown": (item.get("analysis") or {}).get("confidence_breakdown"),
            "source_detections":  item.get("source_detections"),
            "fusion_metadata":    item.get("fusion_metadata"),
        })
    return normalized, top_level_audit


# ---------------------------------------------------------------------------
# Rule summary for the LLM prompt
# ---------------------------------------------------------------------------

def summarize_rules_for_prompt(rules: List[CompiledRule], max_lines: int = 25) -> str:
    """Produce a short human-readable summary of the compiled rules for the LLM."""
    by_category: Dict[str, List[str]] = {}
    for r in rules:
        if not r.human_readable:
            continue
        by_category.setdefault(r.category, []).append(
            f"  - [{r.severity}] {r.human_readable}"
        )
    parts = []
    for cat in ("presence", "spatial", "relational", "subtype"):
        if cat not in by_category:
            continue
        parts.append(f"{cat.upper()} RULES:")
        parts.extend(by_category[cat][:max_lines])
    return "\n".join(parts) if parts else "(no rule summary available)"


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

class BatchRunner:
    """Orchestrates end-to-end evaluation for a dataset."""

    def __init__(self,
                 compiled_rules: List[CompiledRule],
                 evaluator: RuleEvaluator,
                 subtype_resolver: SubtypeResolver,
                 final_scorer: FinalScorer,
                 llm_validator: Optional[LLMValidator] = None):
        self.rules = compiled_rules
        self.evaluator = evaluator
        self.subtype_resolver = subtype_resolver
        self.final_scorer = final_scorer
        self.llm_validator = llm_validator

    def run(self,
            ensemble_path: str,
            captions_path: Optional[str] = None,
            image_col: str = "Image_path",
            caption_col: str = "Image_caption",
            verbose: bool = True) -> Dict[str, Any]:
        """
        Execute the full pipeline. Returns a dict with metadata, statistics, and
        per-image results. Caller is responsible for writing the output to disk.

        The returned dict includes any fusion audit trail the ensemble JSON
        carried (fusion_metadata at the top, per-image fusion_details /
        agreement_analysis inside each result). Fields are omitted silently
        when the input ensemble JSON did not emit them.
        """
        if verbose:
            print("=" * 70)
            print("PCMDE BATCH EVALUATION")
            print("=" * 70)
            print(f"Ensemble JSON: {ensemble_path}")
            print(f"Captions CSV:  {captions_path or '(none)'}")
            print()

        # 1. Load ensemble (returns normalized per-image list and any
        #    top-level audit fields the ensemble script emitted)
        ensemble_items, top_level_audit = load_ensemble_json(ensemble_path)
        if verbose:
            print(f"Loaded {len(ensemble_items)} ensemble results")

        # 2. Load captions (optional)
        captions: Dict[str, str] = {}
        if captions_path:
            captions = load_captions_csv(
                captions_path, image_col=image_col, caption_col=caption_col, verbose=verbose
            )

        # 3. ID matching report
        ensemble_ids = {it["image_id"] for it in ensemble_items}
        caption_ids = set(captions.keys())
        matched = ensemble_ids & caption_ids
        missing_captions = ensemble_ids - caption_ids
        extra_captions = caption_ids - ensemble_ids
        match_rate = (len(matched) / len(ensemble_ids) * 100.0) if ensemble_ids else 0.0
        if verbose and captions:
            print(f"Caption match rate: {match_rate:.1f}% ({len(matched)}/{len(ensemble_ids)})")
            if missing_captions:
                print(f"  {len(missing_captions)} images have no caption")
            if extra_captions:
                print(f"  {len(extra_captions)} captions have no corresponding image")
            print()

        # 4. Rule summary for LLM prompts
        rule_summary = summarize_rules_for_prompt(self.rules)

        # 5. Per-image evaluation
        results: List[FinalScore] = []
        per_image_audit: Dict[str, Dict[str, Any]] = {}

        for idx, item in enumerate(ensemble_items, start=1):
            image_id = item["image_id"]
            caption = captions.get(image_id, "")
            if verbose:
                print(f"[{idx}/{len(ensemble_items)}] {image_id}", end=" ... ", flush=True)

            subtype = self.subtype_resolver.resolve(caption=caption, filename=image_id)

            rule_eval = self.evaluator.evaluate(
                image_id=image_id,
                fused_counts=item["fused_counts"],
                detector_boxes=item["detector_boxes"],
                subtype=subtype,
                ensemble_confidence=item.get("ensemble_confidence"),
            )

            if self.llm_validator is not None:
                from .llm_validator import LLMValidationResult
                try:
                    llm_eval = self.llm_validator.validate(
                        caption=caption,
                        subtype=subtype,
                        fused_counts=item["fused_counts"],
                        detector_boxes=item["detector_boxes"],
                        ensemble_confidence=rule_eval.ensemble_confidence,
                        rule_summary=rule_summary,
                    )
                except Exception as exc:
                    llm_eval = LLMValidationResult(
                        score=0.0, verdict="FAIL", reasoning="",
                        subtype_display_name=subtype.display_name,
                        failed=True, failure_reason=f"validator error: {exc}",
                    )
                if llm_eval.failed:
                    # No usable LLM judgement. Mirror the rule score (as the
                    # --no-llm path does) so the final score is not moved by an
                    # invented value, and flag the image so it can be re-run
                    # before any number is reported.
                    if verbose:
                        print(f"LLM FAILED ({llm_eval.failure_reason}) ... ", end="")
                    llm_eval = LLMValidationResult(
                        score=rule_eval.overall_rule_score,
                        verdict=rule_eval.pass_fail,
                        reasoning=f"(LLM failed: {llm_eval.failure_reason})",
                        subtype_display_name=subtype.display_name,
                        confidence=0.0,
                        failed=True,
                        failure_reason=llm_eval.failure_reason,
                    )
            else:
                from .llm_validator import LLMValidationResult
                llm_eval = LLMValidationResult(
                    score=rule_eval.overall_rule_score,  # mirror rule score when LLM disabled
                    verdict=rule_eval.pass_fail,
                    reasoning="(LLM validator disabled)",
                    subtype_display_name=subtype.display_name,
                    confidence=0.0,
                )

            final = self.final_scorer.combine(rule_eval, llm_eval)
            results.append(final)

            # Stash per-image fusion audit for the output metadata. Only
            # keep keys the ensemble actually emitted; drop None entries.
            audit_entry = {
                k: v for k, v in {
                    "fusion_details":       item.get("fusion_details"),
                    "agreement_analysis":   item.get("agreement_analysis"),
                    "confidence_breakdown": item.get("confidence_breakdown"),
                    "source_detections":    item.get("source_detections"),
                    "fusion_metadata":      item.get("fusion_metadata"),
                }.items() if v is not None
            }
            if audit_entry:
                per_image_audit[image_id] = audit_entry

            if verbose:
                marker = "PASS" if final.verdict == "PASS" else "FAIL"
                print(f"{marker} (score={final.overall_score:.1f}, "
                      f"rule={final.rule_score:.1f}, llm={final.llm_score:.1f}, "
                      f"subtype={final.subtype_display_name})")

        # 6. Aggregate statistics
        stats = _compute_stats(results)
        if self.llm_validator is not None:
            stats["llm_failures"] = sum(1 for r in results if r.llm_failed)
        if verbose:
            print()
            print("=" * 70)
            print("SUMMARY")
            print("=" * 70)
            for k, v in stats.items():
                print(f"  {k}: {v}")
        if stats.get("llm_failures"):
            # Printed even in --quiet mode: these scores must not be reported.
            failed_ids = [r.image_id for r in results if r.llm_failed]
            print()
            print(f"WARNING: {len(failed_ids)} image(s) got no valid LLM reply; their LLM "
                  "score mirrors the rule score (llm_failed=true in the output).")
            print("         Fix the LLM setup (timeout, max_tokens, server) and re-run "
                  "before reporting results:")
            print("         " + ", ".join(failed_ids))

        output = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "ensemble_path": ensemble_path,
                "captions_path": captions_path,
                "n_rules": len(self.rules),
                "llm_enabled": self.llm_validator is not None,
                "match_rate": match_rate,
                "images_without_caption": sorted(missing_captions) if captions else [],
            },
            "statistics": stats,
            "results": [asdict(r) for r in results],
        }

        # Fusion audit trail (optional, present only if the ensemble JSON
        # emitted it). Populated under metadata so the evaluation output
        # stays focused on scoring while preserving full traceability for
        # downstream debugging or analysis.
        if top_level_audit:
            output["metadata"]["ensemble_top_level_audit"] = top_level_audit
        if per_image_audit:
            output["metadata"]["per_image_fusion_audit"] = per_image_audit

        return output


def _compute_stats(results: List[FinalScore]) -> Dict[str, Any]:
    if not results:
        return {
            "total": 0, "passed": 0, "failed": 0, "pass_rate": 0.0,
            "mean_overall": 0.0, "mean_rule": 0.0, "mean_llm": 0.0,
        }
    n = len(results)
    passed = sum(1 for r in results if r.verdict == "PASS")
    mean_overall = sum(r.overall_score for r in results) / n
    mean_rule = sum(r.rule_score for r in results) / n
    mean_llm = sum(r.llm_score for r in results) / n
    min_overall = min(r.overall_score for r in results)
    max_overall = max(r.overall_score for r in results)
    subtype_counts: Dict[str, int] = {}
    for r in results:
        subtype_counts[r.subtype_display_name] = subtype_counts.get(r.subtype_display_name, 0) + 1
    return {
        "total": n,
        "passed": passed,
        "failed": n - passed,
        "pass_rate": round(passed / n * 100.0, 2),
        "mean_overall": round(mean_overall, 2),
        "mean_rule": round(mean_rule, 2),
        "mean_llm": round(mean_llm, 2),
        "min_overall": round(min_overall, 2),
        "max_overall": round(max_overall, 2),
        "subtype_counts": subtype_counts,
    }
