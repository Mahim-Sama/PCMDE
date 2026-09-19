"""
Rule loader (v2).

Loads a self-describing rules JSON and compiles it into a list of
CompiledRule objects that the evaluator can execute.

The rules JSON is expected to follow the canonical structured schema:

    {
      "object_type": "<string>",
      "subtypes": { "<id>": {...}, ... },
      "fallback_subtype": { "subtype_id": "...", "display_name": "..." },
      "scoring_weights": {...},
      "pass_threshold": <number>,
      "rules": [
        {
          "rule_type": "<one of 10 rule types>",
          "subject_component": "<string, optional for critical_presence>",
          "reference_component": "<string, optional>",
          "parameters": {...},
          "severity": "critical|major|minor",
          "description": "<string>",
          "applies_to_subtypes": ["<id>", ...]  (optional)
        }, ...
      ]
    }

This replaces the previous two-file approach (rules JSON + interpreter
vocabulary JSON). Every rule is now self-contained: its rule_type maps
directly to a predicate, its parameters are already in the shape the
predicate expects, and its applicability is expressed inline.

Unknown rule types raise a clear error at load time rather than being
silently skipped, so domain authors (or the LLM that generates the rules)
get immediate feedback.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Data types (unchanged from v1)
# ---------------------------------------------------------------------------

@dataclass
class CompiledRule:
    """
    An executable rule produced by the loader.

    Attributes:
        rule_id:        Stable identifier for diagnostics.
        category:       One of "presence", "spatial", "relational", "subtype".
        severity:       "critical" | "major" | "minor".
        predicate:      Name of the predicate from predicates.py to invoke.
        params:         Parameters to pass to the predicate.
        subject:        Component name this rule operates on, or None.
        reference:      Reference component name, or None.
        applies_when:   Applicability condition. Supported keys:
                            {"always": True}
                            {"subtype_in":     ["id", ...]}
                            {"subtype_not_in": ["id", ...]}
        human_readable: Short description from the rules JSON.
    """
    rule_id: str
    category: str
    severity: str
    predicate: str
    params: Dict[str, Any] = field(default_factory=dict)
    subject: Optional[str] = None
    reference: Optional[str] = None
    applies_when: Dict[str, Any] = field(default_factory=lambda: {"always": True})
    human_readable: str = ""


class RuleCompilationError(ValueError):
    """Raised when a rule in the JSON cannot be compiled."""


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

class RuleLoader:
    """
    Compiles a structured rules JSON into CompiledRule objects.
    """

    SUPPORTED_RULE_TYPES = {
        "count_constraint",
        "containment_constraint",
        "forbid_overlap",
        "attachment_constraint",
        "relative_position_constraint",
        "size_ratio_constraint",
        "size_hierarchy",
        "dependency_constraint",
        "subtype_restriction",
        "critical_presence",
    }

    def __init__(self, rules_path: Path):
        self.rules_path = Path(rules_path)
        if not self.rules_path.exists():
            raise FileNotFoundError(f"Rules JSON not found: {self.rules_path}")
        with open(self.rules_path, "r", encoding="utf-8") as fh:
            self.rules_json: Dict[str, Any] = json.load(fh)

        if "rules" not in self.rules_json:
            raise RuleCompilationError(
                f"Rules JSON at {self.rules_path} is missing the top-level 'rules' list."
            )

    # ------------------------------------------------------------------
    # Convenience accessors (the pipeline reads these from here instead
    # of juggling separate knowledge/vocab files)
    # ------------------------------------------------------------------

    def get_object_type(self) -> str:
        return self.rules_json.get("object_type", "unknown")

    def get_subtypes(self) -> Dict[str, Dict[str, Any]]:
        return self.rules_json.get("subtypes", {})

    def get_fallback_subtype(self) -> Dict[str, str]:
        fb = self.rules_json.get("fallback_subtype", {})
        return {
            "subtype_id": fb.get("subtype_id", "generic"),
            "display_name": fb.get("display_name", "Generic"),
        }

    def get_subtype_notes(self) -> Dict[str, str]:
        return self.rules_json.get("subtype_notes", {})

    def get_scoring_weights(self) -> Dict[str, float]:
        return self.rules_json.get("scoring_weights", {
            "presence": 0.30, "spatial": 0.35,
            "relational": 0.20, "subtype": 0.15,
        })

    def get_pass_threshold(self) -> Optional[float]:
        thresh = self.rules_json.get("pass_threshold")
        return float(thresh) if thresh is not None else None

    def get_llm_prompt_template(self) -> Optional[str]:
        return self.rules_json.get("llm_prompt_template")

    # ------------------------------------------------------------------
    # Main compile entry point
    # ------------------------------------------------------------------

    def compile(self) -> List[CompiledRule]:
        """
        Compile every entry under "rules" into a CompiledRule.

        Raises RuleCompilationError with a clear message if any rule cannot
        be compiled (unknown rule_type, missing required fields, etc.).
        """
        compiled: List[CompiledRule] = []
        for idx, rule in enumerate(self.rules_json["rules"]):
            try:
                compiled.append(self._compile_one(rule, idx))
            except RuleCompilationError:
                raise
            except Exception as exc:
                raise RuleCompilationError(
                    f"Failed to compile rule at index {idx}: {exc}"
                ) from exc
        return compiled

    # ------------------------------------------------------------------
    # Per-rule compilation: one big dispatch on rule_type
    # ------------------------------------------------------------------

    def _compile_one(self, rule: Dict[str, Any], idx: int) -> CompiledRule:
        rule_type = rule.get("rule_type")
        if rule_type is None:
            raise RuleCompilationError(f"Rule at index {idx} is missing 'rule_type'.")
        if rule_type not in self.SUPPORTED_RULE_TYPES:
            raise RuleCompilationError(
                f"Rule at index {idx} uses unsupported rule_type '{rule_type}'. "
                f"Supported: {sorted(self.SUPPORTED_RULE_TYPES)}."
            )

        severity = rule.get("severity", "major")
        description = rule.get("description", "")
        subject = rule.get("subject_component")
        reference = rule.get("reference_component")
        params = dict(rule.get("parameters") or {})

        applies_to = rule.get("applies_to_subtypes")
        applies_when = (
            {"subtype_in": list(applies_to)}
            if applies_to else {"always": True}
        )

        # Dispatch
        dispatcher = {
            "count_constraint":             self._compile_count_constraint,
            "containment_constraint":       self._compile_containment,
            "forbid_overlap":               self._compile_forbid_overlap,
            "attachment_constraint":        self._compile_attachment,
            "relative_position_constraint": self._compile_relative_position,
            "size_ratio_constraint":        self._compile_size_ratio,
            "size_hierarchy":               self._compile_size_hierarchy,
            "dependency_constraint":        self._compile_dependency,
            "subtype_restriction":          self._compile_subtype_restriction,
            "critical_presence":            self._compile_critical_presence,
        }
        return dispatcher[rule_type](
            rule_type, idx, severity, description,
            subject, reference, params, applies_when,
        )

    # ------------------------------------------------------------------
    # Per-rule-type compilers. Each returns a CompiledRule.
    # ------------------------------------------------------------------

    def _compile_count_constraint(self, rule_type, idx, severity, desc,
                                   subject, reference, params, applies_when):
        if subject is None:
            raise RuleCompilationError(
                f"count_constraint at index {idx}: subject_component is required."
            )
        # Mode A: exact_count
        if "exact_count" in params:
            return CompiledRule(
                rule_id=f"count_exact.{subject}.{idx}",
                category="presence",
                severity=severity,
                predicate="count_exact",
                params={"expected": int(params["exact_count"])},
                subject=subject,
                applies_when=applies_when,
                human_readable=desc,
            )
        # Mode B: count_range
        if "count_range" in params:
            rng = params["count_range"]
            return CompiledRule(
                rule_id=f"count_range.{subject}.{idx}",
                category="presence",
                severity=severity,
                predicate="count_in_range",
                params={"min": int(rng["min"]), "max": int(rng["max"])},
                subject=subject,
                applies_when=applies_when,
                human_readable=desc,
            )
        # Mode C: count_by_subtype. We fan this out into one rule per subtype,
        # each applying only when the image is that subtype. To avoid changing
        # the return type, we emit the first one here and store the rest on
        # a sidecar attribute; the compile() wrapper will handle this.
        # Simpler approach: return a special "multi" marker by raising an
        # internal signal -- but that complicates things. Instead, we build
        # an ad-hoc multi-rule and let compile() expand it.
        if "count_by_subtype" in params:
            # Return a sentinel; compile() will expand it.
            return _CountBySubtypeExpansion(
                idx=idx, severity=severity, desc=desc,
                subject=subject,
                counts_map=dict(params["count_by_subtype"]),
                applies_when=applies_when,
            )
        raise RuleCompilationError(
            f"count_constraint at index {idx} has none of "
            f"'exact_count', 'count_range', 'count_by_subtype'."
        )

    def _compile_containment(self, rule_type, idx, severity, desc,
                              subject, reference, params, applies_when):
        self._require_subject_and_reference(rule_type, idx, subject, reference)
        return CompiledRule(
            rule_id=f"containment.{subject}_in_{reference}.{idx}",
            category="spatial",
            severity=severity,
            predicate="containment",
            params={
                "min_ratio": float(params.get("min_containment_ratio", 0.7)),
                "require_all": True,
            },
            subject=subject,
            reference=reference,
            applies_when=applies_when,
            human_readable=desc,
        )

    def _compile_forbid_overlap(self, rule_type, idx, severity, desc,
                                 subject, reference, params, applies_when):
        self._require_subject_and_reference(rule_type, idx, subject, reference)
        return CompiledRule(
            rule_id=f"forbid_overlap.{subject}_on_{reference}.{idx}",
            category="spatial",
            severity=severity,
            predicate="forbid_containment",
            params={
                "max_ratio": float(params.get("max_overlap_ratio", 0.3)),
                "min_reference_overlap": float(params.get("min_reference_overlap", 0.0)),
            },
            subject=subject,
            reference=reference,
            applies_when=applies_when,
            human_readable=desc,
        )

    def _compile_attachment(self, rule_type, idx, severity, desc,
                             subject, reference, params, applies_when):
        self._require_subject_and_reference(rule_type, idx, subject, reference)
        return CompiledRule(
            rule_id=f"attachment.{subject}_to_{reference}.{idx}",
            category="spatial",
            severity=severity,
            predicate="require_proximity",
            params={"margin_fraction": float(params.get("margin_fraction", 0.25))},
            subject=subject,
            reference=reference,
            applies_when=applies_when,
            human_readable=desc,
        )

    def _compile_relative_position(self, rule_type, idx, severity, desc,
                                    subject, reference, params, applies_when):
        if subject is None:
            raise RuleCompilationError(
                f"relative_position_constraint at index {idx}: subject_component is required."
            )
        axis = params.get("axis")
        if axis not in ("x", "y"):
            raise RuleCompilationError(
                f"relative_position_constraint at index {idx}: 'axis' must be 'x' or 'y'."
            )
        tolerance_px = float(params.get("tolerance_px", 0.0))
        # Scale-invariant tolerance. The predicates give it precedence over
        # tolerance_px, so it must be forwarded whenever the rule declares it.
        tolerance_fraction = (
            {"tolerance_fraction": float(params["tolerance_fraction"])}
            if "tolerance_fraction" in params else {}
        )

        # Mode B: extremum (reference omitted, position supplied)
        if "position" in params and reference is None:
            position = params["position"]
            if position not in ("min", "max", "either"):
                raise RuleCompilationError(
                    f"relative_position_constraint at index {idx}: "
                    f"'position' must be 'min', 'max', or 'either'."
                )
            return CompiledRule(
                rule_id=f"extremum.{subject}.{axis}_{position}.{idx}",
                category="spatial",
                severity=severity,
                predicate="at_extremum",
                params={"axis": axis, "position": position,
                        "tolerance_px": float(params.get("tolerance_px", 50.0)),
                        **tolerance_fraction},
                subject=subject,
                applies_when=applies_when,
                human_readable=desc,
            )

        # Mode A: relative to reference
        if reference is None:
            raise RuleCompilationError(
                f"relative_position_constraint at index {idx}: "
                f"reference_component is required when 'position' is not given."
            )
        direction = params.get("direction")
        valid_directions = {
            "above_centroid", "below_centroid",
            "left_of_centroid", "right_of_centroid",
            "opposite",
        }
        if direction not in valid_directions:
            raise RuleCompilationError(
                f"relative_position_constraint at index {idx}: "
                f"'direction' must be one of {sorted(valid_directions)}."
            )

        # below_centroid on axis y: use the specialized below_centroid predicate
        if direction == "below_centroid" and axis == "y":
            return CompiledRule(
                rule_id=f"below_centroid.{subject}_vs_{reference}.{idx}",
                category="spatial",
                severity=severity,
                predicate="below_centroid",
                params={"offset_fraction": 0.0, "require_all": True},
                subject=subject,
                reference=reference,
                applies_when=applies_when,
                human_readable=desc,
            )

        if direction == "above_centroid" and axis == "y":
            return CompiledRule(
                rule_id=f"above_or_lateral.{subject}_vs_{reference}.{idx}",
                category="spatial",
                severity=severity,
                predicate="above_or_lateral_centroid",
                params={"lateral_margin_fraction": 0.2, "require_all": True},
                subject=subject,
                reference=reference,
                applies_when=applies_when,
                human_readable=desc,
            )

        # All other (axis, direction) pairs use the generic relative_position primitive.
        primitive_direction = {
            "above_centroid":     "less",
            "below_centroid":     "greater",
            "left_of_centroid":   "less",
            "right_of_centroid":  "greater",
            "opposite":           "opposite",
        }[direction]
        return CompiledRule(
            rule_id=f"relpos.{subject}_vs_{reference}.{axis}_{direction}.{idx}",
            category="spatial",
            severity=severity,
            predicate="relative_position",
            params={"axis": axis, "direction": primitive_direction,
                    "tolerance_px": tolerance_px, "require_all": True,
                    **tolerance_fraction},
            subject=subject,
            reference=reference,
            applies_when=applies_when,
            human_readable=desc,
        )

    def _compile_size_ratio(self, rule_type, idx, severity, desc,
                             subject, reference, params, applies_when):
        self._require_subject_and_reference(rule_type, idx, subject, reference)
        dim = params.get("dimension", "width")
        if dim not in ("width", "height", "area"):
            raise RuleCompilationError(
                f"size_ratio_constraint at index {idx}: "
                f"'dimension' must be 'width', 'height', or 'area'."
            )

        # Mode B: bounds_by_subtype -- emit one rule per overridden subtype,
        # plus one default rule for everyone else. Mirrors the count_by_subtype
        # fan-out pattern. The default min_ratio/max_ratio are still required:
        # they are used by the fallback rule for subtypes not in the override
        # map.
        if "bounds_by_subtype" in params:
            if "min_ratio" not in params or "max_ratio" not in params:
                raise RuleCompilationError(
                    f"size_ratio_constraint at index {idx}: when using "
                    f"'bounds_by_subtype', top-level 'min_ratio' and 'max_ratio' "
                    f"are still required as the fallback for unspecified subtypes."
                )
            return _SizeRatioBoundsBySubtypeExpansion(
                idx=idx, severity=severity, desc=desc,
                subject=subject, reference=reference, dim=dim,
                default_min=float(params["min_ratio"]),
                default_max=float(params["max_ratio"]),
                bounds_map=dict(params["bounds_by_subtype"]),
                applies_when=applies_when,
            )

        # Mode A: simple global bounds (legacy)
        return CompiledRule(
            rule_id=f"size_ratio.{subject}_{dim}_over_{reference}_{dim}.{idx}",
            category="spatial",
            severity=severity,
            predicate="size_ratio_in_range",
            params={
                "dimension": dim,
                "min_ratio": float(params["min_ratio"]),
                "max_ratio": float(params["max_ratio"]),
                "aggregation": "max",
            },
            subject=subject,
            reference=reference,
            applies_when=applies_when,
            human_readable=desc,
        )

    def _compile_size_hierarchy(self, rule_type, idx, severity, desc,
                                 subject, reference, params, applies_when):
        self._require_subject_and_reference(rule_type, idx, subject, reference)
        dim = params.get("dimension", "area")
        if dim not in ("width", "height", "area"):
            raise RuleCompilationError(
                f"size_hierarchy at index {idx}: "
                f"'dimension' must be 'width', 'height', or 'area'."
            )
        return CompiledRule(
            rule_id=f"size_hierarchy.{reference}_gt_{subject}_{dim}.{idx}",
            category="relational",
            severity=severity,
            predicate="size_hierarchy",
            params={
                "dimension": dim,
                "strict": True,
                "tolerance_factor": float(params.get("tolerance_factor", 1.0)),
            },
            subject=subject,
            reference=reference,
            applies_when=applies_when,
            human_readable=desc,
        )

    def _compile_dependency(self, rule_type, idx, severity, desc,
                             subject, reference, params, applies_when):
        self._require_subject_and_reference(rule_type, idx, subject, reference)
        return CompiledRule(
            rule_id=f"dependency.{subject}_requires_{reference}.{idx}",
            category="relational",
            severity=severity,
            predicate="requires_component",
            params={},
            subject=subject,
            reference=reference,
            applies_when=applies_when,
            human_readable=desc,
        )

    def _compile_subtype_restriction(self, rule_type, idx, severity, desc,
                                      subject, reference, params, applies_when):
        if subject is None:
            raise RuleCompilationError(
                f"subtype_restriction at index {idx}: subject_component is required."
            )
        if "allowed_for_subtypes" in params:
            allowed = list(params["allowed_for_subtypes"])
            return CompiledRule(
                rule_id=f"subtype_restrict.{subject}.allowed.{idx}",
                category="subtype",
                severity=severity,
                predicate="count_exact",
                params={"expected": 0},
                subject=subject,
                applies_when={"subtype_not_in": allowed},
                human_readable=desc,
            )
        if "forbidden_for_subtypes" in params:
            forbidden = list(params["forbidden_for_subtypes"])
            return CompiledRule(
                rule_id=f"subtype_restrict.{subject}.forbidden.{idx}",
                category="subtype",
                severity=severity,
                predicate="count_exact",
                params={"expected": 0},
                subject=subject,
                applies_when={"subtype_in": forbidden},
                human_readable=desc,
            )
        raise RuleCompilationError(
            f"subtype_restriction at index {idx} requires one of "
            f"'allowed_for_subtypes' or 'forbidden_for_subtypes'."
        )

    def _compile_critical_presence(self, rule_type, idx, severity, desc,
                                    subject, reference, params, applies_when):
        components = list(params.get("required_components") or [])
        if not components:
            raise RuleCompilationError(
                f"critical_presence at index {idx}: "
                f"parameters.required_components must be a non-empty list."
            )
        return CompiledRule(
            rule_id=f"critical_presence.{'_'.join(components)}.{idx}",
            category="presence",
            severity="critical",
            predicate="require_all_present",
            params={"components": components},
            applies_when=applies_when,
            human_readable=desc,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _require_subject_and_reference(rule_type, idx, subject, reference):
        if subject is None:
            raise RuleCompilationError(
                f"{rule_type} at index {idx}: subject_component is required."
            )
        if reference is None:
            raise RuleCompilationError(
                f"{rule_type} at index {idx}: reference_component is required."
            )


# ---------------------------------------------------------------------------
# Sentinel for count_by_subtype expansion
# ---------------------------------------------------------------------------

@dataclass
class _CountBySubtypeExpansion:
    """
    Placeholder returned by _compile_count_constraint when the rule uses
    count_by_subtype. It is expanded into one CompiledRule per subtype by
    RuleLoader.compile() wrapper logic below.
    """
    idx: int
    severity: str
    desc: str
    subject: str
    counts_map: Dict[str, int]
    applies_when: Dict[str, Any]


@dataclass
class _SizeRatioBoundsBySubtypeExpansion:
    """
    Placeholder returned by _compile_size_ratio when the rule uses
    bounds_by_subtype. It is expanded into one CompiledRule per overridden
    subtype plus one default rule for the unspecified subtypes.
    """
    idx: int
    severity: str
    desc: str
    subject: str
    reference: str
    dim: str
    default_min: float
    default_max: float
    bounds_map: Dict[str, Dict[str, float]]
    applies_when: Dict[str, Any]


# Override compile() to post-process expansions
_original_compile = RuleLoader.compile


def _compile_with_expansion(self) -> List[CompiledRule]:
    raw = _original_compile(self)
    expanded: List[CompiledRule] = []
    for item in raw:
        if isinstance(item, _CountBySubtypeExpansion):
            exp = item
            # Intersect explicit applies_to_subtypes (if any) with the
            # subtypes present in the counts_map. The rule is per-subtype.
            base_applies_in = None
            if "subtype_in" in exp.applies_when:
                base_applies_in = set(exp.applies_when["subtype_in"])
            for subtype_id, count in exp.counts_map.items():
                if base_applies_in is not None and subtype_id not in base_applies_in:
                    continue
                expanded.append(CompiledRule(
                    rule_id=f"count_exact.{exp.subject}.{subtype_id}.{exp.idx}",
                    category="presence",
                    severity=exp.severity,
                    predicate="count_exact",
                    params={"expected": int(count)},
                    subject=exp.subject,
                    applies_when={"subtype_in": [subtype_id]},
                    human_readable=f"{exp.desc} [{subtype_id}={count}]",
                ))
        elif isinstance(item, _SizeRatioBoundsBySubtypeExpansion):
            exp = item
            base_applies_in = None
            if "subtype_in" in exp.applies_when:
                base_applies_in = set(exp.applies_when["subtype_in"])

            # Per-subtype rules with overridden bounds
            for subtype_id, bounds in exp.bounds_map.items():
                if base_applies_in is not None and subtype_id not in base_applies_in:
                    continue
                expanded.append(CompiledRule(
                    rule_id=(f"size_ratio.{exp.subject}_{exp.dim}_over_"
                             f"{exp.reference}_{exp.dim}.{subtype_id}.{exp.idx}"),
                    category="spatial",
                    severity=exp.severity,
                    predicate="size_ratio_in_range",
                    params={
                        "dimension": exp.dim,
                        "min_ratio": float(bounds["min_ratio"]),
                        "max_ratio": float(bounds["max_ratio"]),
                        "aggregation": "max",
                    },
                    subject=exp.subject,
                    reference=exp.reference,
                    applies_when={"subtype_in": [subtype_id]},
                    human_readable=(
                        f"{exp.desc} [{subtype_id}: "
                        f"{bounds['min_ratio']}-{bounds['max_ratio']}]"
                    ),
                ))

            # Fallback rule with default bounds for the remaining subtypes
            override_subtypes = list(exp.bounds_map.keys())
            if base_applies_in is not None:
                fallback_subtypes = [s for s in base_applies_in
                                     if s not in override_subtypes]
                fallback_applies_when = (
                    {"subtype_in": fallback_subtypes}
                    if fallback_subtypes else None
                )
            else:
                fallback_applies_when = {"subtype_not_in": override_subtypes}

            if fallback_applies_when is not None:
                expanded.append(CompiledRule(
                    rule_id=(f"size_ratio.{exp.subject}_{exp.dim}_over_"
                             f"{exp.reference}_{exp.dim}.default.{exp.idx}"),
                    category="spatial",
                    severity=exp.severity,
                    predicate="size_ratio_in_range",
                    params={
                        "dimension": exp.dim,
                        "min_ratio": exp.default_min,
                        "max_ratio": exp.default_max,
                        "aggregation": "max",
                    },
                    subject=exp.subject,
                    reference=exp.reference,
                    applies_when=fallback_applies_when,
                    human_readable=(
                        f"{exp.desc} [default: "
                        f"{exp.default_min}-{exp.default_max}]"
                    ),
                ))
        else:
            expanded.append(item)
    return expanded


RuleLoader.compile = _compile_with_expansion  # type: ignore[assignment]