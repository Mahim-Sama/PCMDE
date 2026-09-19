"""
Domain-agnostic LLM validator.

Renders a prompt from a Jinja-like template string loaded from the domain's
knowledge JSON, calls the LLM, and parses the structured response.

The prompt template uses simple {placeholder} substitution (Python str.format),
not full Jinja, to avoid an extra dependency. This keeps the framework
minimal while still allowing per-domain customization of prompts.

Expected fields in the LLM response (same structure across all domains):

    SCORE: 0-100
    VERDICT: PASS or FAIL
    SUBTYPE: <subtype display name>
    CONFIDENCE: 0.0-1.0
    SPECIFICATION_VIOLATIONS: <list or None>
    SPATIAL_VIOLATIONS: <list or None>
    REASONING: <free-form 3-5 sentences>
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .llm_client import OllamaClient
from .subtype_resolver import SubtypeResolution


@dataclass
class LLMValidationResult:
    """Structured result from LLM validation."""
    score: float
    verdict: str
    reasoning: str
    subtype_display_name: str
    spec_violations: List[str] = field(default_factory=list)
    spatial_violations: List[str] = field(default_factory=list)
    confidence: float = 0.5
    raw_response: str = ""
    # True when no attempt produced a parseable SCORE. The score is then NOT an
    # LLM judgement; the batch runner substitutes the rule score and flags it.
    failed: bool = False
    failure_reason: str = ""
    valid_runs: int = 0


# deepseek-r1 and other reasoning models prefix the answer with a <think> block.
# A block that never closes means the reply was cut off while still thinking.
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_UNCLOSED_THINK = re.compile(r"<think>.*\Z", re.IGNORECASE | re.DOTALL)
_SCORE = re.compile(r"SCORE\s*:\s*\**\s*(\d+(?:\.\d+)?)", re.IGNORECASE)


def strip_reasoning(text: str) -> str:
    """Remove <think>...</think> blocks (and an unterminated trailing one)."""
    text = _THINK_BLOCK.sub("", text or "")
    return _UNCLOSED_THINK.sub("", text).strip()


def extract_score(text: str) -> Optional[float]:
    """SCORE value from the answer part of a reply, or None if absent."""
    m = _SCORE.search(strip_reasoning(text))
    return float(m.group(1)) if m else None


class LLMValidator:
    """
    Composes a prompt from a template, invokes the LLM, and parses the reply.
    """

    def __init__(self,
                 client: OllamaClient,
                 prompt_template: str,
                 subtype_notes: Optional[Dict[str, str]] = None,
                 runs_per_image: int = 3,
                 max_attempts: int = 6,
                 variance_threshold: float = 10.0):
        """
        Args:
            client: OllamaClient or compatible.
            prompt_template: str.format template with placeholders for
                {subtype}, {subtype_id}, {caption}, {counts_json},
                {boxes_json}, {ensemble_confidence}, {rule_summary},
                {subtype_notes}.
            subtype_notes: optional mapping from subtype_id -> per-subtype
                note string. Injected into the prompt under {subtype_notes}
                when the current image's subtype matches. Pulled directly
                from the rules JSON's "subtype_notes" block by the pipeline
                factory; no separate knowledge file required.
        """
        self.client = client
        self.prompt_template = prompt_template
        self._subtype_notes: Dict[str, str] = dict(subtype_notes or {})
        self.runs_per_image = max(1, int(runs_per_image))
        self.max_attempts = max(self.runs_per_image, int(max_attempts))
        self.variance_threshold = float(variance_threshold)

    # def _stable_generate(self, prompt: str, n_runs: int = 3, variance_threshold: float = 10.0) -> str:
    #     scores = []
    #     responses = []
    #     for _ in range(n_runs):
    #         resp = self.client.generate(prompt)
    #         score = float(re.search(r"SCORE:\s*(\d+(?:\.\d+)?)", resp, re.IGNORECASE).group(1))
    #         scores.append(score)
    #         responses.append(resp)
        
    #     if max(scores) - min(scores) > variance_threshold:
    #         # Fallback to median response text
    #         median_idx = scores.index(sorted(scores)[len(scores)//2])
    #         return responses[median_idx]
    #     return responses[0]

    def _stable_generate(self, prompt: str) -> Tuple[Optional[str], int, List[str]]:
        """
        Paper Sec. III-D: query the LLM three times; if the scores spread by
        more than the variance threshold, keep the median run.

        Only replies that contain a parseable SCORE count as runs. Failed calls
        (timeouts, HTTP errors, empty or truncated replies) are retried up to
        max_attempts in total instead of being scored as a made-up neutral
        value, which previously gave silent 50/70 scores.

        Returns (chosen_response or None, number_of_valid_runs, failure_notes).
        """
        scores: List[float] = []
        responses: List[str] = []
        failures: List[str] = []
        attempts = 0
        while len(scores) < self.runs_per_image and attempts < self.max_attempts:
            attempts += 1
            try:
                resp = self.client.generate(prompt)
            except Exception as exc:
                failures.append(f"attempt {attempts}: {exc}")
                continue
            score = extract_score(resp)
            if score is None:
                reason = getattr(self.client, "last_done_reason", None)
                failures.append(
                    f"attempt {attempts}: no SCORE in reply"
                    + (f" (done_reason={reason})" if reason else "")
                    + ("" if (resp or "").strip() else " (empty reply)")
                )
                continue
            scores.append(score)
            responses.append(resp)

        if not scores:
            return None, 0, failures
        if max(scores) - min(scores) > self.variance_threshold:
            median_idx = scores.index(sorted(scores)[len(scores) // 2])
            return responses[median_idx], len(scores), failures
        return responses[0], len(scores), failures

    def validate(self,
                 caption: str,
                 subtype: SubtypeResolution,
                 fused_counts: Dict[str, int],
                 detector_boxes: Dict[str, List[Dict]],
                 ensemble_confidence: float,
                 rule_summary: str) -> LLMValidationResult:
        """
        Run the LLM validation pass for one image.

        Args:
            caption: caption or free-text description of the image.
            subtype: resolved subtype (species, aircraft type, etc.)
            fused_counts: ensemble counts
            detector_boxes: ensemble bboxes keyed by component. On-disk these
                are stored under "yolo_detections"; renamed here because the
                detector is not required to be YOLO.
            ensemble_confidence: image-level ensemble confidence
            rule_summary: short textual summary of domain rules to include in prompt
        """
        subtype_notes = self._get_subtype_notes(subtype.subtype_id)

        prompt = self.prompt_template.format(
            subtype=subtype.display_name,
            subtype_id=subtype.subtype_id,
            caption=caption or "(no caption provided)",
            counts_json=json.dumps(fused_counts, indent=2),
            boxes_json=json.dumps(detector_boxes, indent=2) if detector_boxes else "(no bounding boxes)",
            ensemble_confidence=f"{ensemble_confidence:.2f}",
            rule_summary=rule_summary,
            subtype_notes=subtype_notes,
        )

        # response_text = self.client.generate(prompt)
        # return self._parse_response(response_text, subtype.display_name)
        
        # # 1. Run LLM 3 times (Sec 3.4, Eq 8)
        # scores = []
        # responses = []
        # for _ in range(3):
        #     resp = self.client.generate(prompt)
        #     # Extract score safely for variance check
        #     score_m = re.search(r"SCORE:\s*(\d+(?:\.\d+)?)", resp, re.IGNORECASE)
        #     scores.append(float(score_m.group(1)) if score_m else 70.0)
        #     responses.append(resp)
        
        # # 2. Dispersion check: variance > 10 → fallback to median run
        # # (Using range as a robust proxy for 3-point variance; matches paper's intent)
        # dispersion = max(scores) - min(scores)
        # if dispersion > 10:
        #     # Take the run closest to the median score
        #     median_val = sorted(scores)[1]
        #     final_response = responses[scores.index(median_val)]
        # else:
        #     final_response = responses[0]  # All consistent, use first

        final_response, valid_runs, failures = self._stable_generate(prompt)
        if final_response is None:
            return LLMValidationResult(
                score=0.0,
                verdict="FAIL",
                reasoning="",
                subtype_display_name=subtype.display_name,
                confidence=0.0,
                failed=True,
                failure_reason="; ".join(failures) or "no valid LLM reply",
            )
        result = self._parse_response(final_response, subtype.display_name)
        result.valid_runs = valid_runs
        return result

    def _get_subtype_notes(self, subtype_id: str) -> str:
        """
        Look up per-subtype instructions to inject into the prompt (e.g.,
        "this aircraft legitimately has tail-mounted engines" or "chickens
        have beaks and wings"). The mapping is supplied by the pipeline
        factory from the rules JSON's "subtype_notes" block.
        """
        note = self._subtype_notes.get(subtype_id)
        if not note:
            return ""
        return f"\nSUBTYPE NOTE: {note}\n"

    # @staticmethod
    # def _parse_response(text: str, default_subtype_name: str) -> LLMValidationResult:
    #     """
    #     Parse the structured LLM output. Missing fields fall back to sensible defaults.
    #     """
    #     def find(key, pattern_tail=r"\s*(.+?)(?=\n[A-Z_]+:|$)"):
    #         m = re.search(rf"{key}:{pattern_tail}", text, re.IGNORECASE | re.DOTALL)
    #         return m.group(1).strip() if m else ""

    #     # Score
    #     score_match = re.search(r"SCORE:\s*(\d+(?:\.\d+)?)", text, re.IGNORECASE)
    #     score = float(score_match.group(1)) if score_match else 70.0
    #     score = max(0.0, min(100.0, score))

    #     # Verdict
    #     v_match = re.search(r"VERDICT:\s*(PASS|FAIL)", text, re.IGNORECASE)
    #     verdict = v_match.group(1).upper() if v_match else ("PASS" if score >= 60 else "FAIL")

    #     # Subtype display
    #     subtype_name = find("SUBTYPE") or default_subtype_name

    #     # Confidence
    #     c_match = re.search(r"CONFIDENCE:\s*(\d+(?:\.\d+)?)", text, re.IGNORECASE)
    #     confidence = float(c_match.group(1)) if c_match else 0.7
    #     confidence = max(0.0, min(1.0, confidence))

    #     # Violation lists
    #     spec_block = find("SPECIFICATION_VIOLATIONS")
    #     spatial_block = find("SPATIAL_VIOLATIONS")
    #     spec_violations = _parse_bullet_list(spec_block)
    #     spatial_violations = _parse_bullet_list(spatial_block)

    #     # Reasoning (always the last block if no other fields follow)
    #     reasoning_match = re.search(r"REASONING:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    #     reasoning = reasoning_match.group(1).strip() if reasoning_match else text[-400:]
    #     reasoning = re.sub(r"\s+", " ", reasoning)[:600]

    #     return LLMValidationResult(
    #         score=score,
    #         verdict=verdict,
    #         reasoning=reasoning,
    #         subtype_display_name=subtype_name,
    #         spec_violations=spec_violations,
    #         spatial_violations=spatial_violations,
    #         confidence=confidence,
    #         raw_response=text,
    #     )

    @staticmethod
    def _parse_response(text: str, default_subtype_name: str) -> LLMValidationResult:
        # 1. Drop reasoning-model <think> blocks, then markdown code fences.
        #    Otherwise a "SCORE:" echoed while thinking can be parsed as the answer.
        clean_text = strip_reasoning(text)
        clean_text = re.sub(r"```(?:json)?\s*|\s*```", "", clean_text, flags=re.IGNORECASE).strip()
        
        def extract_field(key, pattern=r"\s*(.+?)(?=\n[A-Z_]+:|$)"):
            m = re.search(rf"{key}\s*:\s*{pattern}", clean_text, re.IGNORECASE | re.DOTALL)
            return m.group(1).strip() if m else ""
    
        # 2. Score: tolerate decimals, extra text, or markdown artifacts.
        #    Callers only pass replies whose SCORE was found (see _stable_generate);
        #    the 70.0 default is kept for direct callers of this parser.
        score_match = _SCORE.search(clean_text)
        score = float(score_match.group(1)) if score_match else 70.0
        score = max(0.0, min(100.0, score))
    
        # 3. Verdict
        v_match = re.search(r"VERDICT\s*:\s*(PASS|FAIL)", clean_text, re.IGNORECASE)
        verdict = v_match.group(1).upper() if v_match else ("PASS" if score >= 60 else "FAIL")
    
        # 4. Subtype & Confidence
        subtype_name = extract_field("SUBTYPE") or default_subtype_name
        c_match = re.search(r"CONFIDENCE\s*:\s*(\d+(?:\.\d+)?)", clean_text, re.IGNORECASE)
        confidence = float(c_match.group(1)) if c_match else 0.7
        confidence = max(0.0, min(1.0, confidence))
    
        # 5. Violations & Reasoning
        spec_block = extract_field("SPECIFICATION_VIOLATIONS")
        spatial_block = extract_field("SPATIAL_VIOLATIONS")
        spec_violations = _parse_bullet_list(spec_block)
        spatial_violations = _parse_bullet_list(spatial_block)
    
        reasoning_match = re.search(r"REASONING\s*:\s*(.+)", clean_text, re.IGNORECASE | re.DOTALL)
        reasoning = reasoning_match.group(1).strip() if reasoning_match else clean_text[-400:]
        reasoning = re.sub(r"\s+", " ", reasoning)[:600]
    
        return LLMValidationResult(
            score=score, verdict=verdict, reasoning=reasoning,
            subtype_display_name=subtype_name,
            spec_violations=spec_violations, spatial_violations=spatial_violations,
            confidence=confidence, raw_response=text,
        )

def _parse_bullet_list(block: str) -> List[str]:
    """Parse a text block into a list of items, splitting on newlines and bullets."""
    if not block:
        return []
    stripped = block.strip()
    if stripped.lower() in ("none", "none found", "n/a", "na", "[]", ""):
        return []
    items = []
    for line in stripped.split("\n"):
        cleaned = line.strip().lstrip("-*").strip()
        if cleaned:
            items.append(cleaned)
    return items