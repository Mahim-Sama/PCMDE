"""
Pipeline factory (v2).

Reads a single YAML config file and builds a fully-wired BatchRunner.
The domain's rules JSON is now self-describing and carries everything
previously split across three files (rules + interpreter_vocab + knowledge).

Adding a new domain: write ONE rules JSON and ONE config YAML. No Python
changes; no vocabulary file; no knowledge file.
"""

from pathlib import Path
from typing import Any, Dict, Optional

from .batch_runner import BatchRunner
from .config_manager import ConfigError, ConfigManager
from .evaluator import RuleEvaluator
from .final_scorer import FinalScorer
from .llm_client import LLMClientConfig, OllamaClient
from .llm_validator import LLMValidator
from .rule_loader import RuleLoader
from .subtype_resolver import SubtypeResolver

CANONICAL_CATEGORIES = {"presence", "spatial", "relational", "subtype"}

def build_pipeline(config: ConfigManager) -> BatchRunner:
    """
    Construct a fully-configured BatchRunner from a ConfigManager instance.
    """
    # --- Load and compile rules ---
    rules_path = Path(config.require("paths.rules_file"))
    loader = RuleLoader(rules_path=rules_path)
    compiled_rules = loader.compile()

    # --- Category weights: prefer the rules JSON, fall back to config ---
    weights = loader.get_scoring_weights()
    if not weights:
        weights = config.get("scoring.category_weights", {
            "presence": 0.30,
            "spatial": 0.35,
            "relational": 0.20,
            "subtype": 0.15,
        })
    # Normalize any aliased key names to canonical category labels
    # (presence / spatial / relational / subtype). If a new alias shows up in
    # a future rules JSON, add it here or the weight will be treated as a
    # separate category.
    weight_aliases = {
        "presence_accuracy": "presence",
        "presence_count": "presence",
        "spatial_alignment": "spatial",
        "relational_consistency": "relational",
        "species_logic_match": "subtype",
        "caption_alignment": "subtype",
    }
    canonical_weights: Dict[str, float] = {}
    for k, v in weights.items():
        key = weight_aliases.get(k, k)
        canonical_weights[key] = canonical_weights.get(key, 0.0) + float(v)

    unknown_cats = set(canonical_weights.keys()) - CANONICAL_CATEGORIES
    if unknown_cats:
        raise ConfigError(
            f"Unknown scoring categories in rules JSON: {unknown_cats}. "
            f"Must be one of: {CANONICAL_CATEGORIES}"
        )

    total = sum(canonical_weights.values())
    if abs(total - 1.0) > 0.05:
        print(f"WARNING: Scoring weights sum to {total:.3f} (expected ~1.0). Renormalization will apply.")
    
    # --- Pass threshold: rules JSON takes precedence, config may override ---
    rules_pass_threshold = loader.get_pass_threshold()
    if rules_pass_threshold is None:
        rules_pass_threshold = 60.0
    pass_threshold = float(config.get("scoring.pass_threshold", rules_pass_threshold))
    component_threshold = float(config.get("scoring.component_threshold", 40.0))
    critical_cap = float(config.get("scoring.critical_violation_cap", 30.0))
    unevaluable_is_violation = bool(config.get("scoring.unevaluable_is_violation", True))

    # --- Rule evaluator ---
    evaluator = RuleEvaluator(
        compiled_rules=compiled_rules,
        category_weights=canonical_weights,
        pass_threshold=pass_threshold,
        critical_violation_cap=critical_cap,
        unevaluable_is_violation=unevaluable_is_violation,
    )

    # --- Subtype resolver: patterns come from the rules JSON itself ---
    subtype_resolver = SubtypeResolver(
        subtypes=loader.get_subtypes(),
        fallback_subtype=loader.get_fallback_subtype(),
    )

    # --- LLM validator (optional) ---
    llm_validator: Optional[LLMValidator] = None
    if config.get("llm.enabled", True):
        llm_cfg = LLMClientConfig(
            model_name=config.get("llm.model_name", "deepseek-r1:70b"),
            api_base=config.get("llm.api_base", "http://localhost:11434"),
            temperature=float(config.get("llm.temperature", 0.0)),
            timeout_seconds=int(config.get("llm.timeout_seconds", 120)),
            max_tokens=int(config.get("llm.max_tokens", 600)),
        )
        client = OllamaClient(llm_cfg)

        prompt_template = loader.get_llm_prompt_template()
        if not prompt_template:
            prompt_template = _default_prompt_template()

        llm_validator = LLMValidator(
            client=client,
            prompt_template=prompt_template,
            subtype_notes=loader.get_subtype_notes(),
            runs_per_image=int(config.get("llm.runs_per_image", 3)),
            max_attempts=int(config.get("llm.max_attempts", 6)),
            variance_threshold=float(config.get("llm.variance_threshold", 10.0)),
        )

    # --- Final scorer ---
    llm_weight = float(config.get("scoring.llm_weight", 0.4))
    rule_weight = float(config.get("scoring.rule_weight", 0.6))
    final_scorer = FinalScorer(
        llm_weight=llm_weight,
        rule_weight=rule_weight,
        pass_threshold=pass_threshold,
        component_threshold=component_threshold,
    )

    return BatchRunner(
        compiled_rules=compiled_rules,
        evaluator=evaluator,
        subtype_resolver=subtype_resolver,
        final_scorer=final_scorer,
        llm_validator=llm_validator,
    )


def _default_prompt_template() -> str:
    """
    Generic fallback prompt template used only when the domain's rules JSON
    does not supply its own 'llm_prompt_template' field. Most domains should
    author a template in their rules JSON so the LLM gets appropriate domain
    framing (zoology expert, aerospace engineer, etc.).
    """
    return (
        "You are a domain expert evaluating a generated image.\n"
        "SUBTYPE: {subtype}\n"
        "CAPTION: \"{caption}\"\n"
        "{subtype_notes}\n"
        "DETECTED COMPONENTS:\n{counts_json}\n\n"
        "BOUNDING BOXES:\n{boxes_json}\n\n"
        "Detection confidence (image-level): {ensemble_confidence}\n\n"
        "RULES BEING CHECKED:\n{rule_summary}\n\n"
        "Assess whether the detected configuration is physically plausible\n"
        "for the stated subtype. Focus on SPECIFICATION matches (correct counts\n"
        "and types of components for this subtype) and SPATIAL relationships\n"
        "(components in physically valid positions).\n\n"
        "RESPOND IN EXACTLY THIS FORMAT:\n\n"
        "SCORE: [0-100]\n"
        "VERDICT: [PASS if >=60, FAIL if <60]\n"
        "SUBTYPE: {subtype}\n"
        "CONFIDENCE: [0.0-1.0]\n"
        "SPECIFICATION_VIOLATIONS: [list or None]\n"
        "SPATIAL_VIOLATIONS: [list or None]\n"
        "REASONING: [3-5 sentences]\n"
    )