"""
Stage 1c: fuse per-image component counts from one object detector and two
vision-language models (VLMs) into the ensemble JSON consumed by the rule
engine (cli/evaluate.py).

Output schema (unchanged):

    {
      "fusion_metadata": {...},                       # provenance of this file
      "results": [
        {"image_id": "...",
         "ensemble_output": {
            "fused_counts":    {"<component>": <int>, ...},
            "confidence":      <float>,
            "yolo_detections": {"<component>": [{x_center, y_center, width,
                                                 height, confidence}, ...]}}}
      ]
    }

Modes
-----
  full           detector + VLM-1 + VLM-2, four-case voting (below). Default.
  detector_only  detector counts and boxes as-is (ablation "CDSD-only /
                 Zero-Shot detection").
  vlm_only       VLM-1 + VLM-2 fused by the same weighted-confidence vote,
                 no bounding boxes (ablation "VLM-only detection").

Voting in full mode (per component):
  (i)   all three agree                    -> that count
  (ii)  detector agrees with either VLM    -> detector count
  (iii) VLMs agree, detector differs       -> detector wins only if
                                              conf_det * w_det > threshold
  (iv)  all disagree                       -> highest conf * weight wins

Roles
-----
The detector is the file whose results carry bounding boxes (or
--detector-file). VLM-1 and VLM-2 are the remaining files IN COMMAND-LINE
ORDER, so the weights (default 0.32 / 0.18) follow the order you pass them.

Component names
---------------
Models spell the same part differently ("Wheels", "wheels", "bonnet (front
hood)", "swept wing"). Every name is normalised (lowercase, parenthetical
glosses dropped, spaces/hyphens -> underscores) and then mapped through the
optional "component_aliases" block of the domain's rules JSON, so all sources
and the rules speak one vocabulary. Within one model's output, spellings of
the same name are merged by max; distinct names mapped onto one canonical
component ("front bumper" + "rear bumper" -> "bumper") are summed.

Image confidence is read from "average_confidence", falling back to
"confidence" (the key the YOLO notebook and some VLM scripts write), then 0.5.

Usage
-----
    # everything from the domain config (inputs, rules aliases, output path)
    python pipeline/ensemble.py --config configs/car.yaml

    # explicit files
    python pipeline/ensemble.py det.json vlm1.json vlm2.json \
        --rules domains/car/physical_rules_car.json -o ensemble.json

    # ablation inputs
    python pipeline/ensemble.py --config configs/car.yaml --mode vlm_only -o vlm_only.json
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

MODES = ("full", "detector_only", "vlm_only")
DEFAULT_WEIGHTS = {"detector": 0.50, "vlm1": 0.32, "vlm2": 0.18}
DEFAULT_DETECTOR_THRESHOLD = 0.30
CONFIDENCE_KEYS = ("average_confidence", "confidence")


# ---------------------------------------------------------------------------
# Component vocabulary
# ---------------------------------------------------------------------------

_PARENTHETICAL = re.compile(r"\([^)]*\)")
_SEPARATORS = re.compile(r"[\s\-]+")


def normalize_component_name(name: str) -> str:
    """'Rearview_Mirrors' -> 'rearview_mirrors', 'bonnet (front hood)' -> 'bonnet'."""
    text = _PARENTHETICAL.sub(" ", str(name)).strip().lower()
    text = _SEPARATORS.sub("_", text)
    return re.sub(r"_+", "_", text).strip("_")


class ComponentVocabulary:
    """Maps raw model labels onto the canonical component names used by the rules."""

    def __init__(self, aliases: Optional[Dict[str, str]] = None, normalize: bool = True):
        self.normalize = normalize
        self.aliases = {
            self._norm(k): self._norm(v) for k, v in (aliases or {}).items()
        }

    @classmethod
    def from_rules_file(cls, path: Optional[str]) -> "ComponentVocabulary":
        if not path:
            return cls()
        rules = load_json(path)
        return cls(rules.get("component_aliases", {}))

    def _norm(self, name: str) -> str:
        return normalize_component_name(name) if self.normalize else str(name)

    def canonical(self, raw: str) -> str:
        name = self._norm(raw)
        return self.aliases.get(name, name)

    def canonical_counts(self, counts: Dict[str, int]) -> Dict[str, int]:
        by_spelling: Dict[str, int] = {}
        for raw, value in (counts or {}).items():
            name = self._norm(raw)
            by_spelling[name] = max(by_spelling.get(name, 0), _as_count(value))
        merged: Dict[str, int] = {}
        for name, value in by_spelling.items():
            canonical = self.aliases.get(name, name)
            merged[canonical] = merged.get(canonical, 0) + value
        return merged


def _as_count(value) -> int:
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return 0


# ---------------------------------------------------------------------------
# Per-source view of one image
# ---------------------------------------------------------------------------

def image_confidence(result: Dict, keys: Sequence[str] = CONFIDENCE_KEYS,
                     default: float = 0.5) -> Tuple[float, str]:
    """Image-level confidence and the key it was read from ("default" if none)."""
    for key in keys:
        value = result.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value), key
    return default, "default"


class SourceImage:
    """One model's output for one image, in canonical component names."""

    def __init__(self, result: Dict, vocab: ComponentVocabulary,
                 confidence_keys: Sequence[str] = CONFIDENCE_KEYS):
        self.image_id = result["image_id"]
        self.counts = vocab.canonical_counts(result.get("components", {}))
        self.confidence, self.confidence_key = image_confidence(result, confidence_keys)
        self.boxes: Dict[str, List[Dict]] = {}
        per_class: Dict[str, List[float]] = {}
        for det in result.get("detections") or []:
            comp = vocab.canonical(det.get("class", "unknown"))
            x1, y1, x2, y2 = det["bbox"]
            self.boxes.setdefault(comp, []).append({
                "x_center": (x1 + x2) / 2,
                "y_center": (y1 + y2) / 2,
                "width": x2 - x1,
                "height": y2 - y1,
                "confidence": det["confidence"],
            })
            per_class.setdefault(comp, []).append(det["confidence"])
        self.component_confidence = {c: sum(v) / len(v) for c, v in per_class.items()}


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------

class EnsembleFusion:
    """Dataset-agnostic fusion of one detector and two VLMs."""

    def __init__(
        self,
        detector_weight: float = DEFAULT_WEIGHTS["detector"],
        vlm1_weight: float = DEFAULT_WEIGHTS["vlm1"],
        vlm2_weight: float = DEFAULT_WEIGHTS["vlm2"],
        detector_threshold: float = DEFAULT_DETECTOR_THRESHOLD,
        vocabulary: Optional[ComponentVocabulary] = None,
        confidence_keys: Sequence[str] = CONFIDENCE_KEYS,
    ):
        self.weights = {"detector": detector_weight, "vlm1": vlm1_weight, "vlm2": vlm2_weight}
        self.detector_threshold = detector_threshold
        self.vocabulary = vocabulary or ComponentVocabulary()
        self.confidence_keys = tuple(confidence_keys)

    # ------------------------------------------------------------------
    # File-level entry point
    # ------------------------------------------------------------------

    def fuse(self, mode: str, detector_data: Optional[Dict],
             vlm1_data: Optional[Dict], vlm2_data: Optional[Dict]) -> Tuple[List[Dict], Dict]:
        """Fuse every image. Returns (results, report)."""
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        needs_detector = mode in ("full", "detector_only")
        needs_vlms = mode in ("full", "vlm_only")
        if needs_detector and detector_data is None:
            raise ValueError(f"mode {mode!r} needs a detector file (one with bounding boxes)")
        if needs_vlms and (vlm1_data is None or vlm2_data is None):
            raise ValueError(f"mode {mode!r} needs two VLM files")

        def index(data):
            return {r["image_id"]: r for r in data["results"]} if data else {}

        det_index, vlm1_index, vlm2_index = index(detector_data), index(vlm1_data), index(vlm2_data)
        driver = (detector_data if needs_detector else vlm1_data)["results"]

        per_image, skipped = [], []
        conf_keys = {"detector": {}, "vlm1": {}, "vlm2": {}}
        for det_or_vlm1 in driver:
            image_id = det_or_vlm1["image_id"]
            sources = {
                "detector": det_index.get(image_id) if needs_detector else None,
                "vlm1": vlm1_index.get(image_id) if needs_vlms else None,
                "vlm2": vlm2_index.get(image_id) if needs_vlms else None,
            }
            missing = [name for name, needed in (("vlm1", needs_vlms), ("vlm2", needs_vlms))
                       if needed and sources[name] is None]
            if missing:
                skipped.append({"image_id": image_id, "missing_in": missing})
                continue
            views = {name: SourceImage(r, self.vocabulary, self.confidence_keys)
                     for name, r in sources.items() if r is not None}
            for name, view in views.items():
                conf_keys[name][view.confidence_key] = conf_keys[name].get(view.confidence_key, 0) + 1
            per_image.append((image_id, views))

        # Components are the union over every image of every source used, so each
        # fused_counts lists the full vocabulary (0 where absent) and the agreement
        # rate in the confidence is computed over the same component set per image.
        components = sorted({c for _, views in per_image for v in views.values() for c in v.counts})

        results = []
        for image_id, views in per_image:
            if mode == "full":
                fused = self._fuse_full(views["detector"], views["vlm1"], views["vlm2"], components)
            elif mode == "detector_only":
                fused = self._fuse_detector_only(views["detector"], components)
            else:
                fused = self._fuse_vlm_only(views["vlm1"], views["vlm2"], components)
            results.append({"image_id": image_id, "ensemble_output": fused})

        report = {
            "n_images": len(results),
            "skipped_images": skipped,
            "confidence_keys_used": {k: v for k, v in conf_keys.items() if v},
        }
        return results, report

    # ------------------------------------------------------------------
    # Per-image fusion
    # ------------------------------------------------------------------

    def _fuse_full(self, det: SourceImage, v1: SourceImage, v2: SourceImage,
                   components: List[str]) -> Dict:
        fused_counts = {}
        for comp in components:
            fused_counts[comp], _, _ = self._apply_voting_rules(
                det.counts.get(comp, 0), v1.counts.get(comp, 0), v2.counts.get(comp, 0),
                det.component_confidence.get(comp, 0.0), v1.confidence, v2.confidence,
            )
        base = (det.confidence * self.weights["detector"]
                + v1.confidence * self.weights["vlm1"]
                + v2.confidence * self.weights["vlm2"])
        confidence = self._agreement_adjusted(base, [det.counts, v1.counts, v2.counts], components)
        return {"fused_counts": fused_counts, "confidence": round(confidence, 6),
                "yolo_detections": det.boxes}

    def _fuse_detector_only(self, det: SourceImage, components: List[str]) -> Dict:
        return {"fused_counts": {c: det.counts.get(c, 0) for c in components},
                "confidence": round(min(det.confidence, 1.0), 6),
                "yolo_detections": det.boxes}

    def _fuse_vlm_only(self, v1: SourceImage, v2: SourceImage, components: List[str]) -> Dict:
        w1, w2 = self.weights["vlm1"], self.weights["vlm2"]
        fused_counts = {}
        for comp in components:
            a, b = v1.counts.get(comp, 0), v2.counts.get(comp, 0)
            if a == b:
                fused_counts[comp] = a
            else:
                # Same weighted-confidence vote as case (iv), minus the detector.
                fused_counts[comp] = a if v1.confidence * w1 >= v2.confidence * w2 else b
        base = (v1.confidence * w1 + v2.confidence * w2) / (w1 + w2) if (w1 + w2) > 0 else 0.5
        confidence = self._agreement_adjusted(base, [v1.counts, v2.counts], components)
        return {"fused_counts": fused_counts, "confidence": round(confidence, 6),
                "yolo_detections": {}}

    # ------------------------------------------------------------------
    # Voting rules  (Cases i - iv)
    # ------------------------------------------------------------------

    def _apply_voting_rules(
        self,
        D: int,
        V1: int,
        V2: int,
        conf_det: float,
        conf_vlm1: float,
        conf_vlm2: float,
    ) -> Tuple[int, str, str]:
        """
        Returns (fused_count, case_id, reasoning).

        Case (i)   - All agree
        Case (ii)  - Detector agrees with >=1 VLM
        Case (iii) - VLMs agree, detector differs  -> confidence check
        Case (iv)  - All disagree                  -> weighted vote
        """
        # Case (i)
        if D == V1 == V2:
            return D, "case_i", f"All agree: {D}"

        # Case (ii)
        if D == V1 or D == V2:
            return D, "case_ii", f"Detector agrees with a VLM: {D}"

        # Case (iii)
        if V1 == V2:
            y = conf_det * self.weights["detector"]
            if y > self.detector_threshold:
                return D, "case_iii_detector", (
                    f"VLMs={V1}, Detector={D}; "
                    f"detector score {y:.3f} > {self.detector_threshold} -> detector wins"
                )
            return V1, "case_iii_vlm", (
                f"VLMs={V1}, Detector={D}; "
                f"detector score {y:.3f} <= {self.detector_threshold} -> VLMs win"
            )

        # Case (iv)
        score_d = conf_det * self.weights["detector"]
        score_v1 = conf_vlm1 * self.weights["vlm1"]
        score_v2 = conf_vlm2 * self.weights["vlm2"]

        candidates = {"detector": (score_d, D), "vlm1": (score_v1, V1), "vlm2": (score_v2, V2)}
        winner, (wscore, wcount) = max(candidates.items(), key=lambda x: x[1][0])

        return wcount, "case_iv", (
            f"All disagree (D={D},V1={V1},V2={V2}); "
            f"scores D={score_d:.3f} V1={score_v1:.3f} V2={score_v2:.3f}; "
            f"{winner} wins with {wcount}"
        )

    # ------------------------------------------------------------------
    # Ensemble confidence
    # ------------------------------------------------------------------

    @staticmethod
    def _agreement_adjusted(base: float, count_maps: List[Dict[str, int]],
                            components: List[str]) -> float:
        """
        Weighted confidence boosted/penalised by cross-model agreement:
            agreement  = fraction of components on which all sources agree
            multiplier = 1.15 (all) | 1.08 (>=75%) | 1.0 (>=50%) | 0.92 (<50%)
        """
        if not components:
            return min(base, 1.0)
        n_agree = sum(
            1 for c in components
            if len({counts.get(c, 0) for counts in count_maps}) == 1
        )
        agreement_rate = n_agree / len(components)
        if agreement_rate == 1.0:
            multiplier = 1.15
        elif agreement_rate >= 0.75:
            multiplier = 1.08
        elif agreement_rate >= 0.5:
            multiplier = 1.0
        else:
            multiplier = 0.92
        return min(base * multiplier, 1.0)


# ---------------------------------------------------------------------------
# Role assignment
# ---------------------------------------------------------------------------

def _count_results_with_detections(data: Dict) -> int:
    """Count how many results in a file have non-empty bbox `detections`."""
    return sum(1 for r in data.get("results", []) if r.get("detections"))


def identify_roles(
    files: List[Tuple[str, Dict]],
    detector_path: Optional[str] = None,
) -> Tuple[Optional[Tuple[str, Dict]], List[Tuple[str, Dict]]]:
    """
    Return (detector or None, [VLM files in the order given]).

    The detector is the file with the most results carrying bounding boxes,
    or the file named by detector_path. Files without any boxes are VLMs.
    """
    if detector_path is not None:
        wanted = str(Path(detector_path).resolve())
        detector = next((f for f in files if str(Path(f[0]).resolve()) == wanted), None)
        if detector is None:
            raise ValueError(f"--detector-file '{detector_path}' is not among the input files.")
        return detector, [f for f in files if f is not detector]

    with_boxes = sorted(
        ((f, _count_results_with_detections(f[1])) for f in files),
        key=lambda item: item[1], reverse=True,
    )
    if not with_boxes or with_boxes[0][1] == 0:
        return None, list(files)
    if len(with_boxes) > 1 and with_boxes[1][1] == with_boxes[0][1]:
        raise ValueError(
            f"Two files tie for the most bbox detections ({with_boxes[0][1]}). "
            "Use --detector-file to specify which is the object detector."
        )
    detector = with_boxes[0][0]
    return detector, [f for f in files if f is not detector]


# ---------------------------------------------------------------------------
# I/O and orchestration
# ---------------------------------------------------------------------------

def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def save_json(data: Dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    print(f"Saved -> {path}")


def build_ensemble(
    files: Sequence[str],
    mode: str = "full",
    rules_path: Optional[str] = None,
    detector_path: Optional[str] = None,
    detector_weight: float = DEFAULT_WEIGHTS["detector"],
    vlm1_weight: float = DEFAULT_WEIGHTS["vlm1"],
    vlm2_weight: float = DEFAULT_WEIGHTS["vlm2"],
    detector_threshold: float = DEFAULT_DETECTOR_THRESHOLD,
    strict: bool = False,
    verbose: bool = True,
) -> Dict:
    """Load inputs, assign roles, fuse, and return the ensemble JSON document."""
    loaded = [(str(p), load_json(p)) for p in files]
    detector, vlms = identify_roles(loaded, detector_path)
    if mode in ("full", "vlm_only") and len(vlms) != 2:
        raise ValueError(
            f"mode {mode!r} needs exactly two VLM files, found {len(vlms)}: "
            f"{[p for p, _ in vlms]}"
        )
    vlm1 = vlms[0] if len(vlms) > 0 else None
    vlm2 = vlms[1] if len(vlms) > 1 else None

    fusion = EnsembleFusion(
        detector_weight=detector_weight,
        vlm1_weight=vlm1_weight,
        vlm2_weight=vlm2_weight,
        detector_threshold=detector_threshold,
        vocabulary=ComponentVocabulary.from_rules_file(rules_path),
    )
    results, report = fusion.fuse(
        mode,
        detector[1] if detector else None,
        vlm1[1] if (vlm1 and mode != "detector_only") else None,
        vlm2[1] if (vlm2 and mode != "detector_only") else None,
    )

    used = {
        "detector": detector[0] if detector and mode != "vlm_only" else None,
        "vlm1": vlm1[0] if vlm1 and mode != "detector_only" else None,
        "vlm2": vlm2[0] if vlm2 and mode != "detector_only" else None,
    }
    if verbose:
        print(f"Mode            : {mode}")
        for role, path in used.items():
            if path:
                print(f"{role:<16}: {path}")
        print(f"Weights         : detector={detector_weight}, vlm1={vlm1_weight}, vlm2={vlm2_weight}")
        print(f"Aliases from    : {rules_path or '(none - normalisation only)'}")
        for role, keys in report["confidence_keys_used"].items():
            note = "  <-- no confidence field, 0.5 used" if "default" in keys else ""
            print(f"confidence keys : {role} {keys}{note}")

    if report["skipped_images"]:
        msg = (f"{len(report['skipped_images'])} image(s) skipped because a VLM file has no "
               f"entry for them: {[s['image_id'] for s in report['skipped_images']]}")
        if strict:
            raise ValueError(msg)
        print(f"WARNING: {msg}", file=sys.stderr)

    return {
        "fusion_metadata": {
            "mode": mode,
            "inputs": used,
            "weights": {"detector": detector_weight, "vlm1": vlm1_weight, "vlm2": vlm2_weight},
            "detector_threshold": detector_threshold,
            "component_aliases_from": rules_path,
            **report,
        },
        "results": results,
    }


def print_summary(results: List[Dict]) -> None:
    """Print a compact post-run summary."""
    n = len(results)
    if n == 0:
        print("No results to summarise.")
        return

    all_comps: set = set()
    for r in results:
        all_comps.update(r["ensemble_output"]["fused_counts"].keys())

    confidences = [r["ensemble_output"]["confidence"] for r in results]
    print("\n" + "=" * 60)
    print(f"ENSEMBLE COMPLETE  -  {n} images")
    print("=" * 60)
    print(f"Ensemble confidence  avg={sum(confidences) / n:.4f}  "
          f"min={min(confidences):.4f}  max={max(confidences):.4f}")
    print("-" * 60)
    print(f"{'Component':<18}  {'Mean':>6}  {'Min':>4}  {'Max':>4}")
    print("-" * 38)
    for comp in sorted(all_comps):
        vals = [r["ensemble_output"]["fused_counts"].get(comp, 0) for r in results]
        print(f"{comp:<18}  {sum(vals) / n:>6.2f}  {min(vals):>4d}  {max(vals):>4d}")
    print("=" * 60)


def inputs_from_config(config_path: str) -> Tuple[List[str], Optional[str], Optional[str]]:
    """(input files, rules file, default output) declared in a domain config."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from core.config_manager import ConfigManager

    cfg = ConfigManager.load(config_path)
    files = [cfg.get("paths.detections_json")] + list(cfg.get("paths.vlm_jsons") or [])
    return [f for f in files if f], cfg.get("paths.rules_file"), cfg.get("paths.ensemble_json")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Fuse detector + VLM component counts into an ensemble JSON.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("files", nargs="*", metavar="FILE",
                   help="Detector and VLM JSON files. VLM-1/VLM-2 follow command-line order. "
                        "Optional with --config.")
    p.add_argument("--config", help="Domain config: reads paths.detections_json, paths.vlm_jsons, "
                                    "paths.rules_file and (as default output) paths.ensemble_json.")
    p.add_argument("--mode", choices=MODES, default="full")
    p.add_argument("--rules", help="Rules JSON whose component_aliases map model labels to "
                                   "rule component names (default: from --config).")
    p.add_argument("--output", "-o", help="Output path (default: paths.ensemble_json from --config).")
    p.add_argument("--detector-file", help="Declare which input file is the object detector.")
    p.add_argument("--detector-weight", type=float, default=DEFAULT_WEIGHTS["detector"])
    p.add_argument("--vlm1-weight", type=float, default=DEFAULT_WEIGHTS["vlm1"])
    p.add_argument("--vlm2-weight", type=float, default=DEFAULT_WEIGHTS["vlm2"])
    p.add_argument("--detector-threshold", type=float, default=DEFAULT_DETECTOR_THRESHOLD,
                   help="Weighted-confidence threshold for the detector to override "
                        "two agreeing VLMs (case iii).")
    p.add_argument("--strict", action="store_true",
                   help="Fail instead of warning when an image is missing from a VLM file.")
    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)

    files, rules_path, output = list(args.files), args.rules, args.output
    if args.config:
        cfg_files, cfg_rules, cfg_output = inputs_from_config(args.config)
        files = files or cfg_files
        rules_path = rules_path or cfg_rules
        if output is None and args.mode == "full":
            output = cfg_output
    if not files:
        print("Error: no input files (pass FILE arguments or --config).", file=sys.stderr)
        sys.exit(1)
    if not output:
        print("Error: no output path (pass --output).", file=sys.stderr)
        sys.exit(1)

    try:
        document = build_ensemble(
            files, mode=args.mode, rules_path=rules_path, detector_path=args.detector_file,
            detector_weight=args.detector_weight, vlm1_weight=args.vlm1_weight,
            vlm2_weight=args.vlm2_weight, detector_threshold=args.detector_threshold,
            strict=args.strict,
        )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    save_json(document, output)
    print_summary(document["results"])


if __name__ == "__main__":
    main()
