"""
Rule evaluator.

Runs a list of CompiledRule objects against one image's ensemble detections
and produces:

    - per-rule outcomes (pass/fail/unevaluable with diagnostic messages)
    - per-category scores (presence, spatial, relational, subtype)
    - overall weighted score
    - critical-failure detection and score capping

Key design decisions:
    - Unevaluable rules (missing boxes etc.) are treated as violations per the
      user's strict policy, but are clearly tagged so we can distinguish
      detector-holes from real violations in downstream analysis.
    - Critical violations cap the overall rule score at a configurable value.
    - Rule applicability is evaluated based on the resolved subtype.
"""

from dataclasses import dataclass, field
from statistics import mean
from typing import Any, Dict, List, Optional

from .predicates import PredicateResult, get_predicate
from .rule_loader import CompiledRule
from .subtype_resolver import SubtypeResolution


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class RuleOutcome:
    """Outcome of a single rule application against one image."""
    rule_id: str
    category: str
    severity: str
    applicable: bool
    evaluable: bool
    passed: bool
    detail: str
    failure_kind: str = "none"  # one of: none, unevaluable, violation, inapplicable
    human_readable: str = ""


@dataclass
class EvaluationResult:
    """Full evaluation result for one image."""
    image_id: str
    subtype_id: str
    subtype_display_name: str
    subtype_source: str

    category_scores: Dict[str, float] = field(default_factory=dict)
    overall_rule_score: float = 0.0
    pass_fail: str = "FAIL"

    outcomes: List[RuleOutcome] = field(default_factory=list)
    violations: List[str] = field(default_factory=list)
    unevaluable_rules: List[str] = field(default_factory=list)

    ensemble_confidence: float = 0.0

    # Carried through for downstream combining with LLM scores
    fused_counts: Dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _applies_to_subtype(rule: CompiledRule, subtype_id: str) -> bool:
    """Evaluate the rule's applies_when block against a subtype id."""
    cond = rule.applies_when or {}
    if cond.get("always"):
        return True
    if "subtype_in" in cond:
        return subtype_id in cond["subtype_in"]
    if "subtype_not_in" in cond:
        return subtype_id not in cond["subtype_not_in"]
    return True


def _image_level_confidence(detector_boxes: Dict[str, List[Dict]]) -> float:
    """
    Derive an image-level confidence scalar from per-detection confidences.

    This is used only when the ensemble JSON does not provide an explicit
    image-level confidence under ensemble_output.confidence. When that key
    is present (e.g., your ensemble_animal.py emits it going forward), the
    caller will pass its value through directly and this function will not
    be invoked.

    The formula is mean of all per-detection confidences across all
    components. If you prefer min, max, or median aggregation, set it in
    your ensemble script itself and emit the result as
    ensemble_output.confidence.
    """
    vals = []
    for boxes in detector_boxes.values():
        for b in boxes:
            if "confidence" in b:
                vals.append(float(b["confidence"]))
    if not vals:
        return 0.0
    return float(mean(vals))


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class RuleEvaluator:
    """
    Evaluates compiled rules against ensemble output for one image.
    """

    def __init__(self,
                 compiled_rules: List[CompiledRule],
                 category_weights: Dict[str, float],
                 pass_threshold: float = 60.0,
                 critical_violation_cap: float = 30.0,
                 unevaluable_is_violation: bool = True):
        self.rules = compiled_rules
        self.weights = category_weights
        self.pass_threshold = pass_threshold
        self.critical_cap = critical_violation_cap
        self.unevaluable_is_violation = unevaluable_is_violation

    def evaluate(self,
                 image_id: str,
                 fused_counts: Dict[str, int],
                 detector_boxes: Dict[str, List[Dict]],
                 subtype: SubtypeResolution,
                 ensemble_confidence: Optional[float] = None) -> EvaluationResult:
        """
        Evaluate all applicable rules for one image.

        Args:
            image_id: Filename or ID for the image.
            fused_counts: ensemble-level count map, e.g. {"torso": 1, "leg": 4}.
            detector_boxes: ensemble-level bbox map keyed by component, e.g.
                {"torso": [{"x_center": ..., "y_center": ...,
                            "width": ..., "height": ..., "confidence": ...}]}.
                In the on-disk JSON this is stored under the key
                "yolo_detections" (retained for backward compatibility with
                earlier pipeline outputs); internally we call it
                detector_boxes since the detector is not necessarily YOLO.
            subtype: resolved subtype with id and display name.
            ensemble_confidence: image-level confidence emitted by the
                ensemble script under ensemble_output.confidence. When None
                (legacy compact-schema output), it is derived from
                per-detection confidences as a fallback.
        """
        if ensemble_confidence is None:
            ensemble_confidence = _image_level_confidence(detector_boxes)

        outcomes: List[RuleOutcome] = []

        # Flat context list used by predicates like at_extremum
        context_boxes = [b for boxes in detector_boxes.values() for b in boxes]

        any_critical_violation = False

        for rule in self.rules:
            applicable = _applies_to_subtype(rule, subtype.subtype_id)
            if not applicable:
                outcomes.append(RuleOutcome(
                    rule_id=rule.rule_id,
                    category=rule.category,
                    severity=rule.severity,
                    applicable=False,
                    evaluable=True,
                    passed=True,  # inapplicable rules do not count against score
                    detail="inapplicable for subtype",
                    failure_kind="inapplicable",
                    human_readable=rule.human_readable,
                ))
                continue

            # Prepare predicate inputs
            subj_boxes = detector_boxes.get(rule.subject, []) if rule.subject else []
            ref_boxes = detector_boxes.get(rule.reference, []) if rule.reference else []

            params = dict(rule.params)
            # Inject counts when predicates need them
            params["_actual_count"] = fused_counts.get(rule.subject, 0) if rule.subject else 0
            params["_subject_count"] = fused_counts.get(rule.subject, 0) if rule.subject else 0
            params["_reference_count"] = fused_counts.get(rule.reference, 0) if rule.reference else 0
            params["_all_counts"] = fused_counts
            params["_context_boxes"] = context_boxes

            # Vacuous satisfaction for spatial rules whose subject is genuinely absent.
            # If rules say "wings must not overlap head" but the animal has 0 wings,
            # the rule is vacuously true (no wings = no possible overlap). This is
            # distinct from "we can't check" (box missing) which remains unevaluable.
            # Presence rules are NOT short-circuited here: they must still fire when
            # subject count = 0 to detect missing-component violations.
            if (rule.category == "spatial"
                    and rule.subject is not None
                    and params["_subject_count"] == 0):
                outcomes.append(RuleOutcome(
                    rule_id=rule.rule_id,
                    category=rule.category,
                    severity=rule.severity,
                    applicable=True,
                    evaluable=True,
                    passed=True,
                    detail=f"vacuously satisfied: {rule.subject} count = 0",
                    failure_kind="inapplicable",
                    human_readable=rule.human_readable,
                ))
                continue

            # Run predicate
            predicate_fn = get_predicate(rule.predicate)
            try:
                result: PredicateResult = predicate_fn(subj_boxes, ref_boxes, params)
            except Exception as exc:
                result = PredicateResult(
                    passed=False,
                    evaluable=False,
                    detail=f"predicate error: {exc}",
                )

            if not result.evaluable:
                passed_final = False if self.unevaluable_is_violation else True
                failure_kind = "unevaluable" if self.unevaluable_is_violation else "none"
            else:
                passed_final = result.passed
                failure_kind = "none" if result.passed else "violation"

            # The critical cap fires ONLY for evidenced violations, not for
            # unevaluable rules. An unevaluable rule means we lack the bounding
            # box to check; the violation may or may not exist. Firing the cap
            # on unevaluable rules would punish every image for detector holes,
            # not for actual configuration problems. Unevaluable rules still
            # count against the category score (strict policy) but do not
            # activate the catastrophic floor.
            if failure_kind == "violation" and rule.severity == "critical":
                any_critical_violation = True

            outcomes.append(RuleOutcome(
                rule_id=rule.rule_id,
                category=rule.category,
                severity=rule.severity,
                applicable=True,
                evaluable=result.evaluable,
                passed=passed_final,
                detail=result.detail,
                failure_kind=failure_kind,
                human_readable=rule.human_readable,
            ))

        # Compute per-category scores and overall score
        category_scores = self._compute_category_scores(outcomes)
        overall = self._weighted_overall(category_scores)

        if any_critical_violation:
            overall = min(overall, self.critical_cap)

        pass_fail = "PASS" if overall >= self.pass_threshold else "FAIL"

        violations = [
            f"[{o.category}/{o.severity}] {o.rule_id}: {o.detail}"
            for o in outcomes if o.applicable and not o.passed and o.failure_kind == "violation"
        ]
        unevaluable = [
            f"[{o.category}/{o.severity}] {o.rule_id}: {o.detail}"
            for o in outcomes if o.applicable and o.failure_kind == "unevaluable"
        ]

        return EvaluationResult(
            image_id=image_id,
            subtype_id=subtype.subtype_id,
            subtype_display_name=subtype.display_name,
            subtype_source=subtype.source,
            category_scores=category_scores,
            overall_rule_score=overall,
            pass_fail=pass_fail,
            outcomes=outcomes,
            violations=violations,
            unevaluable_rules=unevaluable,
            ensemble_confidence=ensemble_confidence,
            fused_counts=dict(fused_counts),
        )

    # ------------------------------------------------------------------
    # Scoring internals
    # ------------------------------------------------------------------

    def _compute_category_scores(self, outcomes: List[RuleOutcome]) -> Dict[str, float]:
        """
        For each category that has at least one applicable rule, compute:
            (satisfied applicable rules) / (total applicable rules) * 100.

        Categories with zero applicable rules are deliberately OMITTED from
        the result. They are not assigned a default of 100.0. Rationale: the
        weighted average in _weighted_overall renormalizes across only the
        categories present in this dict, so omitting empty categories
        redistributes their weight proportionally to the active ones. This
        matches the paper's rule-score equation where every weighted term
        corresponds to an actually-evaluated category.
        """
        per_category: Dict[str, List[RuleOutcome]] = {}
        for o in outcomes:
            if not o.applicable:
                continue
            per_category.setdefault(o.category, []).append(o)

        scores = {}
        for cat, items in per_category.items():
            n_total = len(items)
            if n_total == 0:
                continue
            n_passed = sum(1 for o in items if o.passed)
            scores[cat] = 100.0 * n_passed / n_total
        return scores

    def _weighted_overall(self, category_scores: Dict[str, float]) -> float:
        """
        Apply configured weights to category scores using redistribution.

        Only categories present in category_scores (i.e., with at least one
        applicable rule for this image) contribute to the weighted sum.
        Their weights are renormalized to sum to 1.0 so that missing
        categories redistribute their share proportionally across the
        active ones rather than silently contributing 100.0 at their
        original weight.

        Example: if weights are {presence: 0.30, spatial: 0.35, relational:
        0.20, subtype: 0.15} and an image has no subtype rules applicable,
        the effective weights become presence=0.353, spatial=0.412,
        relational=0.235 (each divided by 0.85, the sum of active weights).
        """
        active_weights = {
            cat: self.weights.get(cat, 0.0)
            for cat in category_scores.keys()
            if self.weights.get(cat, 0.0) > 0.0
        }
        total_w = sum(active_weights.values())
        if total_w <= 0.0:
            return 0.0

        acc = 0.0
        for cat, score in category_scores.items():
            w = active_weights.get(cat, 0.0)
            if w <= 0.0:
                continue
            acc += (w / total_w) * score
        return acc