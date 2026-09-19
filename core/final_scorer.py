"""
Final scorer.

Fuses rule-based score and LLM-based score into one overall PCMDE score.
Applies dual-threshold PASS criterion: both the overall score and each
component score must meet minimum thresholds.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List

from .evaluator import EvaluationResult
from .llm_validator import LLMValidationResult


@dataclass
class FinalScore:
    image_id: str
    subtype_id: str
    subtype_display_name: str

    overall_score: float
    verdict: str

    rule_score: float
    llm_score: float
    category_scores: Dict[str, float]
    weights: Dict[str, float]

    rule_violations: List[str]
    rule_unevaluable: List[str]
    llm_spec_violations: List[str]
    llm_spatial_violations: List[str]
    llm_reasoning: str

    ensemble_confidence: float
    llm_confidence: float
    combined_confidence: float

    fused_counts: Dict[str, int] = field(default_factory=dict)

    # LLM stage health. llm_failed=True means llm_score is the rule score
    # substituted after every LLM attempt failed, not an LLM judgement.
    llm_failed: bool = False
    llm_failure_reason: str = ""
    llm_valid_runs: int = 0


class FinalScorer:
    """Combines rule + LLM scores per image with configured weights and thresholds."""

    def __init__(self,
                 llm_weight: float = 0.4,
                 rule_weight: float = 0.6,
                 pass_threshold: float = 60.0,
                 component_threshold: float = 40.0):
        if abs(llm_weight + rule_weight - 1.0) > 0.01:
            raise ValueError(
                f"llm_weight + rule_weight must sum to 1.0, got {llm_weight + rule_weight}"
            )
        self.llm_weight = llm_weight
        self.rule_weight = rule_weight
        self.pass_threshold = pass_threshold
        self.component_threshold = component_threshold

    def combine(self,
                rule_eval: EvaluationResult,
                llm_eval: LLMValidationResult) -> FinalScore:
        overall = (self.rule_weight * rule_eval.overall_rule_score
                   + self.llm_weight * llm_eval.score)

        passes = (
            overall >= self.pass_threshold
            and rule_eval.overall_rule_score >= self.component_threshold
            and llm_eval.score >= self.component_threshold
        )
        verdict = "PASS" if passes else "FAIL"

        combined_conf = (rule_eval.ensemble_confidence * 0.6
                         + llm_eval.confidence * 0.4)

        return FinalScore(
            image_id=rule_eval.image_id,
            subtype_id=rule_eval.subtype_id,
            subtype_display_name=rule_eval.subtype_display_name,
            overall_score=overall,
            verdict=verdict,
            rule_score=rule_eval.overall_rule_score,
            llm_score=llm_eval.score,
            category_scores=rule_eval.category_scores,
            weights={
                "rule_weight": self.rule_weight,
                "llm_weight": self.llm_weight,
            },
            rule_violations=rule_eval.violations,
            rule_unevaluable=rule_eval.unevaluable_rules,
            llm_spec_violations=llm_eval.spec_violations,
            llm_spatial_violations=llm_eval.spatial_violations,
            llm_reasoning=llm_eval.reasoning,
            ensemble_confidence=rule_eval.ensemble_confidence,
            llm_confidence=llm_eval.confidence,
            combined_confidence=combined_conf,
            fused_counts=rule_eval.fused_counts,
            llm_failed=llm_eval.failed,
            llm_failure_reason=llm_eval.failure_reason,
            llm_valid_runs=llm_eval.valid_runs,
        )
