"""
Generic predicate primitives.

This module defines the atomic operations that every domain rule ultimately
compiles into. Nothing in this file references aircraft, animals, or any other
specific domain. The entire vocabulary here is geometric, set-theoretic, or
numeric.

A predicate takes three inputs:
    subject_boxes:   list of bbox dicts for the subject component (e.g., "leg")
    reference_boxes: list of bbox dicts for the reference component (e.g., "torso"),
                     or None if the predicate is unary
    params:          dict of predicate-specific parameters (tolerances, thresholds, etc.)

A predicate returns a PredicateResult describing whether the check passed,
which items failed, and whether evaluation was even possible.

Bbox format (single source of truth across the framework):
    {"x_center": float, "y_center": float, "width": float, "height": float, "confidence": float}
    Coordinates are in pixels. Origin is top-left. Y increases downward.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class PredicateResult:
    """
    Result returned by every predicate.

    Attributes:
        passed: Whether the predicate was satisfied. If evaluable is False,
                this field is meaningless on its own.
        evaluable: Whether the predicate could be checked at all. False when
                required bounding boxes are missing from detections.
        detail: Human-readable description of what was checked and what happened.
        item_results: Per-item outcomes for predicates operating on multiple
                items (e.g., multiple legs). Each entry is a dict describing
                one item check.
    """
    passed: bool
    evaluable: bool
    detail: str
    item_results: List[Dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Bounding box helpers (pure geometry, no domain concepts)
# ---------------------------------------------------------------------------

def _bbox_corners(box: Dict[str, float]) -> Tuple[float, float, float, float]:
    """Return (x_min, y_min, x_max, y_max) from a center-format bbox."""
    x_c, y_c = box["x_center"], box["y_center"]
    w, h = box["width"], box["height"]
    half_w, half_h = w / 2.0, h / 2.0
    return x_c - half_w, y_c - half_h, x_c + half_w, y_c + half_h


def _bbox_area(box: Dict[str, float]) -> float:
    return max(0.0, box["width"]) * max(0.0, box["height"])


def _bbox_intersection_area(a: Dict[str, float], b: Dict[str, float]) -> float:
    ax1, ay1, ax2, ay2 = _bbox_corners(a)
    bx1, by1, bx2, by2 = _bbox_corners(b)
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    return iw * ih


def _bbox_iou(a: Dict[str, float], b: Dict[str, float]) -> float:
    inter = _bbox_intersection_area(a, b)
    union = _bbox_area(a) + _bbox_area(b) - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def _bbox_contains_centroid(outer: Dict[str, float], inner: Dict[str, float]) -> bool:
    """Return True if the centroid of 'inner' falls inside 'outer'."""
    ox1, oy1, ox2, oy2 = _bbox_corners(outer)
    return ox1 <= inner["x_center"] <= ox2 and oy1 <= inner["y_center"] <= oy2


def _bbox_inner_containment_ratio(inner: Dict[str, float], outer: Dict[str, float]) -> float:
    """
    Return the fraction of 'inner' that lies inside 'outer'.
    1.0 means fully contained, 0.0 means disjoint.
    """
    inner_area = _bbox_area(inner)
    if inner_area <= 0.0:
        return 0.0
    return _bbox_intersection_area(inner, outer) / inner_area


# ---------------------------------------------------------------------------
# Unary predicates (operate on a single component)
# ---------------------------------------------------------------------------

def count_exact(subject_boxes, reference_boxes, params):
    """
    Check that the number of subject instances equals 'expected'.

    Note: This predicate operates on the COUNT context, not on boxes directly.
    The evaluator passes the counted number via params['_actual_count'].
    """
    expected = params["expected"]
    actual = params.get("_actual_count", len(subject_boxes or []))
    passed = actual == expected
    return PredicateResult(
        passed=passed,
        evaluable=True,
        detail=f"count_exact: expected {expected}, got {actual}",
    )


def count_in_range(subject_boxes, reference_boxes, params):
    """Check that subject count is within [min, max] inclusive."""
    low, high = params["min"], params["max"]
    actual = params.get("_actual_count", len(subject_boxes or []))
    passed = low <= actual <= high
    return PredicateResult(
        passed=passed,
        evaluable=True,
        detail=f"count_in_range: expected [{low},{high}], got {actual}",
    )


# def at_extremum(subject_boxes, reference_boxes, params):
#     """
#     Check that subject occupies an extremum along an axis across all provided
#     context boxes.

#     Parameters:
#         axis: "x" or "y"
#         position: "min", "max", or "either" (subject may be near EITHER the
#             min or max extremum along the axis -- whichever is closer). Use
#             "either" for view-invariant rules: e.g. "head is at one end of
#             the body along x" works whether the animal faces left or right.
#         context_boxes: flat list of all bboxes across all components in the
#             image. Provided by the evaluator via params['_context_boxes'].
#         tolerance_px: pixel tolerance for extremum match.
#     """
#     axis = params["axis"]
#     position = params["position"]
#     tolerance = params.get("tolerance_px", 50.0)
#     context = params.get("_context_boxes", [])

#     if not subject_boxes:
#         return PredicateResult(False, False, "at_extremum: no subject boxes")
#     if not context:
#         return PredicateResult(False, False, "at_extremum: no context boxes")

#     key = "x_center" if axis == "x" else "y_center"
#     all_vals = [b[key] for b in context]
#     subj_val = subject_boxes[0][key]

#     if position == "either":
#         target_min = min(all_vals)
#         target_max = max(all_vals)
#         dev_min = abs(subj_val - target_min)
#         dev_max = abs(subj_val - target_max)
#         if dev_min <= dev_max:
#             target_val, deviation, side = target_min, dev_min, "min"
#         else:
#             target_val, deviation, side = target_max, dev_max, "max"
#         passed = deviation <= tolerance
#         return PredicateResult(
#             passed=passed,
#             evaluable=True,
#             detail=(
#                 f"at_extremum: axis={axis}, position=either, "
#                 f"subject_{key}={subj_val:.1f}, nearer_extremum={target_val:.1f} ({side}), "
#                 f"deviation={deviation:.1f}px, tolerance={tolerance}px"
#             ),
#         )

#     target_val = min(all_vals) if position == "min" else max(all_vals)
#     deviation = abs(subj_val - target_val)
#     passed = deviation <= tolerance
#     return PredicateResult(
#         passed=passed,
#         evaluable=True,
#         detail=(
#             f"at_extremum: axis={axis}, position={position}, "
#             f"subject_{key}={subj_val:.1f}, extremum={target_val:.1f}, "
#             f"deviation={deviation:.1f}px, tolerance={tolerance}px"
#         ),
#     )

def at_extremum(subject_boxes, reference_boxes, params):
    """
    Check that subject occupies an extremum along an axis across all provided
    context boxes.

    Parameters:
        axis: "x" or "y"
        position: "min", "max", or "either"
        context_boxes: provided by evaluator via params['_context_boxes'].
        tolerance_fraction: scale-invariant tolerance, expressed as a fraction
            of the spread (max - min) of context-box centers along the axis.
            Recommended for detector-agnostic rules (boxes from a 1333x800
            GroundingDino resize and a 512x512 YOLO resize give the same
            verdict). Takes precedence over tolerance_px when both are set.
        tolerance_px: legacy raw-pixel tolerance. Default 50.0 if neither
            tolerance_fraction nor tolerance_px is provided.
    """
    axis = params["axis"]
    position = params["position"]
    context = params.get("_context_boxes", [])

    if not subject_boxes:
        return PredicateResult(False, False, "at_extremum: no subject boxes")
    if not context:
        return PredicateResult(False, False, "at_extremum: no context boxes")

    key = "x_center" if axis == "x" else "y_center"
    all_vals = [b[key] for b in context]
    subj_val = subject_boxes[0][key]

    # Resolve tolerance. tolerance_fraction wins if provided.
    if "tolerance_fraction" in params:
        frac = float(params["tolerance_fraction"])
        spread = max(all_vals) - min(all_vals)
        # Guard against a degenerate single-component image where spread=0.
        # Fall back to a small fraction of the subject's own size on that axis.
        if spread <= 0.0:
            size_key = "width" if axis == "x" else "height"
            spread = float(subject_boxes[0].get(size_key, 1.0))
        tolerance = frac * spread
        tol_repr = f"frac={frac:.2f} of spread {spread:.1f}px = {tolerance:.1f}px"
    else:
        tolerance = float(params.get("tolerance_px", 50.0))
        tol_repr = f"{tolerance:.1f}px"

    if position == "either":
        target_min = min(all_vals)
        target_max = max(all_vals)
        dev_min = abs(subj_val - target_min)
        dev_max = abs(subj_val - target_max)
        if dev_min <= dev_max:
            target_val, deviation, side = target_min, dev_min, "min"
        else:
            target_val, deviation, side = target_max, dev_max, "max"
        passed = deviation <= tolerance
        return PredicateResult(
            passed=passed,
            evaluable=True,
            detail=(
                f"at_extremum: axis={axis}, position=either, "
                f"subject_{key}={subj_val:.1f}, nearer_extremum={target_val:.1f} ({side}), "
                f"deviation={deviation:.1f}px, tolerance={tol_repr}"
            ),
        )

    target_val = min(all_vals) if position == "min" else max(all_vals)
    deviation = abs(subj_val - target_val)
    passed = deviation <= tolerance
    return PredicateResult(
        passed=passed,
        evaluable=True,
        detail=(
            f"at_extremum: axis={axis}, position={position}, "
            f"subject_{key}={subj_val:.1f}, extremum={target_val:.1f}, "
            f"deviation={deviation:.1f}px, tolerance={tol_repr}"
        ),
    )

# ---------------------------------------------------------------------------
# Binary spatial predicates (subject vs reference)
# ---------------------------------------------------------------------------

def containment(subject_boxes, reference_boxes, params):
    """
    Check that subject bbox lies inside reference bbox, according to a ratio
    threshold.

    Parameters:
        min_ratio: minimum fraction of subject area that must lie inside
            reference. Default 0.8.
        require_all: if True, every subject must be contained. Default True.
    """
    if not subject_boxes:
        return PredicateResult(False, False, "containment: subject missing")
    if not reference_boxes:
        return PredicateResult(False, False, "containment: reference missing")

    min_ratio = params.get("min_ratio", 0.8)
    require_all = params.get("require_all", True)

    ref = reference_boxes[0]
    items = []
    n_contained = 0
    for i, subj in enumerate(subject_boxes):
        ratio = _bbox_inner_containment_ratio(subj, ref)
        is_contained = ratio >= min_ratio
        if is_contained:
            n_contained += 1
        items.append({
            "index": i,
            "containment_ratio": round(ratio, 3),
            "contained": is_contained,
        })

    passed = (n_contained == len(subject_boxes)) if require_all else (n_contained > 0)
    return PredicateResult(
        passed=passed,
        evaluable=True,
        detail=(
            f"containment: {n_contained}/{len(subject_boxes)} subjects contained "
            f"(min_ratio={min_ratio}, require_all={require_all})"
        ),
        item_results=items,
    )


def forbid_containment(subject_boxes, reference_boxes, params):
    """
    The opposite of containment: subject must NOT lie inside reference.

    Parameters:
        max_ratio: maximum allowed containment ratio (subject area inside
            reference, normalised by subject area) before the rule fires.
            Default 0.3.
        min_reference_overlap: minimum fraction of REFERENCE that the
            intersection must occupy for the rule to actually fire. Default
            0.0 (off; preserves legacy behaviour). Set to a small positive
            value (e.g. 0.15) to make the rule view-tolerant: when a small
            subject bbox falls inside a much larger reference bbox -- because
            the reference bbox includes anatomy attached to the reference
            (trunks, beaks, manes, snouts) -- the intersection is tiny
            relative to the reference's area, so the rule does not fire.
            A genuine "subject is on top of reference" case has the
            intersection occupying a meaningful fraction of BOTH boxes,
            and is correctly flagged.
    """
    if not subject_boxes or not reference_boxes:
        return PredicateResult(False, False, "forbid_containment: boxes missing")

    max_ratio = params.get("max_ratio", 0.3)
    min_ref_overlap = params.get("min_reference_overlap", 0.0)
    ref = reference_boxes[0]
    ref_area = _bbox_area(ref)
    items = []
    n_violating = 0
    for i, subj in enumerate(subject_boxes):
        subj_ratio = _bbox_inner_containment_ratio(subj, ref)
        inter = _bbox_intersection_area(subj, ref)
        ref_ratio = (inter / ref_area) if ref_area > 0 else 0.0
        violates = (subj_ratio > max_ratio) and (ref_ratio >= min_ref_overlap)
        if violates:
            n_violating += 1
        items.append({
            "index": i,
            "containment_ratio": round(subj_ratio, 3),
            "reference_overlap": round(ref_ratio, 3),
            "violates": violates,
        })

    passed = n_violating == 0
    return PredicateResult(
        passed=passed,
        evaluable=True,
        detail=(
            f"forbid_containment: {n_violating}/{len(subject_boxes)} subjects "
            f"improperly inside reference (max_ratio={max_ratio}, "
            f"min_reference_overlap={min_ref_overlap})"
        ),
        item_results=items,
    )


def relative_position(subject_boxes, reference_boxes, params):
    """
    Check subject position relative to reference along one axis.

    Parameters:
        axis: "x" or "y"
        direction: "greater" (subject centroid > reference centroid)
                   "less" (subject centroid < reference centroid)
                   "opposite" (subject on opposite side of image from reference
                               relative to the image midpoint)
        tolerance_px: soft tolerance. Default 0.
        require_all: if True, every subject must satisfy. Default True.
    """
    if not subject_boxes or not reference_boxes:
        return PredicateResult(False, False, "relative_position: boxes missing")

    axis = params["axis"]
    direction = params["direction"]
    ref = reference_boxes[0]

    if "tolerance_fraction" in params:
        frac = float(params["tolerance_fraction"])
        context = params.get("_context_boxes", [])
        ctx_vals = [b["x_center" if axis == "x" else "y_center"] for b in context]
        spread = (max(ctx_vals) - min(ctx_vals)) if len(ctx_vals) >= 2 else 0.0
        if spread <= 0.0:
            size_key = "width" if axis == "x" else "height"
            spread = float(ref.get(size_key, 1.0))
        tolerance = frac * spread
    else:
        tolerance = float(params.get("tolerance_px", 0.0))
    require_all = params.get("require_all", True)

    key = "x_center" if axis == "x" else "y_center"
    ref_val = ref[key]

    items = []
    n_ok = 0
    for i, subj in enumerate(subject_boxes):
        sv = subj[key]
        if direction == "greater":
            ok = sv >= ref_val - tolerance
        elif direction == "less":
            ok = sv <= ref_val + tolerance
        elif direction == "opposite":
            # Treat image midpoint as the reflection axis: subject should be
            # on the opposite side of the ref_val.
            ok = (sv - ref_val) * (ref_val - sv) < 0 or abs(sv - ref_val) > tolerance
            # Simpler reading: subject and reference differ meaningfully.
            ok = abs(sv - ref_val) > tolerance
        else:
            return PredicateResult(False, True, f"relative_position: unknown direction '{direction}'")

        if ok:
            n_ok += 1
        items.append({"index": i, f"subj_{key}": sv, f"ref_{key}": ref_val, "ok": ok})

    passed = (n_ok == len(subject_boxes)) if require_all else (n_ok > 0)
    return PredicateResult(
        passed=passed,
        evaluable=True,
        detail=(
            f"relative_position: axis={axis}, direction={direction}, "
            f"{n_ok}/{len(subject_boxes)} satisfied (tolerance={tolerance}px)"
        ),
        item_results=items,
    )


def below_centroid(subject_boxes, reference_boxes, params):
    """
    Check that subject centroids are below (greater y) the reference centroid.

    Parameters:
        offset_fraction: fraction of reference height that the subject must be
            below the reference centroid. Default 0 (anywhere below centroid
            counts). A value of 0.25 requires subjects to be below the
            centroid by at least a quarter of the reference height.
        require_all: default True.
    """
    if not subject_boxes or not reference_boxes:
        return PredicateResult(False, False, "below_centroid: boxes missing")

    offset_fraction = params.get("offset_fraction", 0.0)
    require_all = params.get("require_all", True)
    ref = reference_boxes[0]
    threshold_y = ref["y_center"] + offset_fraction * ref["height"]

    items = []
    n_ok = 0
    for i, subj in enumerate(subject_boxes):
        ok = subj["y_center"] >= threshold_y
        if ok:
            n_ok += 1
        items.append({
            "index": i,
            "subj_y": subj["y_center"],
            "threshold_y": round(threshold_y, 1),
            "ok": ok,
        })

    passed = (n_ok == len(subject_boxes)) if require_all else (n_ok > 0)
    return PredicateResult(
        passed=passed,
        evaluable=True,
        detail=f"below_centroid: {n_ok}/{len(subject_boxes)} below y={threshold_y:.1f}",
        item_results=items,
    )


def above_or_lateral_centroid(subject_boxes, reference_boxes, params):
    """
    Check that subject centroids are either above (less y) the reference
    centroid, or laterally offset beyond a horizontal margin.

    Used for components that attach to the sides or top of a body part
    (e.g., wings on an avian torso are mid-dorsal or lateral, not ventral).
    """
    if not subject_boxes or not reference_boxes:
        return PredicateResult(False, False, "above_or_lateral_centroid: boxes missing")

    ref = reference_boxes[0]
    lateral_margin_frac = params.get("lateral_margin_fraction", 0.25)
    margin_px = lateral_margin_frac * ref["width"]
    require_all = params.get("require_all", True)

    items = []
    n_ok = 0
    for i, subj in enumerate(subject_boxes):
        above = subj["y_center"] <= ref["y_center"]
        lateral = abs(subj["x_center"] - ref["x_center"]) >= margin_px
        ok = above or lateral
        if ok:
            n_ok += 1
        items.append({
            "index": i,
            "above": above,
            "lateral_offset": round(abs(subj["x_center"] - ref["x_center"]), 1),
            "ok": ok,
        })

    passed = (n_ok == len(subject_boxes)) if require_all else (n_ok > 0)
    return PredicateResult(
        passed=passed,
        evaluable=True,
        detail=f"above_or_lateral_centroid: {n_ok}/{len(subject_boxes)} satisfied",
        item_results=items,
    )


def require_proximity(subject_boxes, reference_boxes, params):
    """
    Check that subject boxes overlap or touch reference (no 'detached'
    components). Uses expanded reference bbox with margin.

    Parameters:
        margin_fraction: enlarge reference bbox by this fraction of its size
            when checking overlap. Default 0.2.
        require_all: default True.
    """
    if not subject_boxes or not reference_boxes:
        return PredicateResult(False, False, "require_proximity: boxes missing")

    margin = params.get("margin_fraction", 0.2)
    require_all = params.get("require_all", True)
    ref = reference_boxes[0]
    expanded_ref = {
        "x_center": ref["x_center"],
        "y_center": ref["y_center"],
        "width": ref["width"] * (1.0 + 2.0 * margin),
        "height": ref["height"] * (1.0 + 2.0 * margin),
    }

    items = []
    n_ok = 0
    for i, subj in enumerate(subject_boxes):
        overlap = _bbox_intersection_area(subj, expanded_ref)
        ok = overlap > 0.0
        if ok:
            n_ok += 1
        items.append({"index": i, "overlap_area": round(overlap, 1), "ok": ok})

    passed = (n_ok == len(subject_boxes)) if require_all else (n_ok > 0)
    return PredicateResult(
        passed=passed,
        evaluable=True,
        detail=f"require_proximity: {n_ok}/{len(subject_boxes)} attached to reference",
        item_results=items,
    )


# ---------------------------------------------------------------------------
# Size and ratio predicates
# ---------------------------------------------------------------------------

def size_ratio_in_range(subject_boxes, reference_boxes, params):
    """
    Check that the size ratio of subject to reference falls in [min_ratio, max_ratio].

    Parameters:
        dimension: "width", "height", or "area"
        min_ratio, max_ratio: bounds on ratio (subject / reference)
        aggregation: "max" (use largest subject), "mean", or "min". Default "max".
    """
    if not subject_boxes or not reference_boxes:
        return PredicateResult(False, False, "size_ratio_in_range: boxes missing")

    dim = params.get("dimension", "width")
    lo, hi = params["min_ratio"], params["max_ratio"]
    agg = params.get("aggregation", "max")

    def measure(b):
        if dim == "area":
            return _bbox_area(b)
        return b[dim]

    ref_val = measure(reference_boxes[0])
    if ref_val <= 0.0:
        return PredicateResult(False, False, "size_ratio_in_range: reference has zero extent")

    subj_vals = [measure(s) for s in subject_boxes]
    if agg == "max":
        subj_val = max(subj_vals)
    elif agg == "min":
        subj_val = min(subj_vals)
    else:
        subj_val = sum(subj_vals) / len(subj_vals)

    ratio = subj_val / ref_val
    passed = lo <= ratio <= hi
    return PredicateResult(
        passed=passed,
        evaluable=True,
        detail=(
            f"size_ratio_in_range: {dim} ratio={ratio:.3f} "
            f"(bounds=[{lo:.3f},{hi:.3f}], agg={agg})"
        ),
    )


def size_hierarchy(subject_boxes, reference_boxes, params):
    """
    Check that subject is smaller than reference along a dimension, with an
    optional multiplicative tolerance to accommodate non-canonical views.

    Parameters:
        dimension: "width", "height", or "area"
        strict: if True, must be strictly less. Default True.
        tolerance_factor: multiplicative slack on the reference. Default 1.0
            (legacy behaviour: subject < reference). With tolerance_factor=2.0
            the rule passes whenever subject < 2 * reference, which is useful
            for hierarchies that hold loosely under foreshortening or
            close-up framing (e.g., head area can exceed torso area in a
            three-quarters view because the torso is foreshortened).
    """
    if not subject_boxes or not reference_boxes:
        return PredicateResult(False, False, "size_hierarchy: boxes missing")

    dim = params.get("dimension", "area")
    strict = params.get("strict", True)
    tolerance_factor = float(params.get("tolerance_factor", 1.0))

    def measure(b):
        if dim == "area":
            return _bbox_area(b)
        return b[dim]

    raw_ref_val = measure(reference_boxes[0])
    threshold = raw_ref_val * tolerance_factor
    items = []
    n_ok = 0
    for i, subj in enumerate(subject_boxes):
        sv = measure(subj)
        ok = sv < threshold if strict else sv <= threshold
        if ok:
            n_ok += 1
        items.append({"index": i, "subj": sv, "ref": raw_ref_val,
                      "threshold": threshold, "ok": ok})

    passed = n_ok == len(subject_boxes)
    return PredicateResult(
        passed=passed,
        evaluable=True,
        detail=(
            f"size_hierarchy: {n_ok}/{len(subject_boxes)} smaller than "
            f"reference*{tolerance_factor} ({dim})"
        ),
        item_results=items,
    )


# ---------------------------------------------------------------------------
# Set-logic predicates (operate on counts only, no geometry)
# ---------------------------------------------------------------------------

def requires_component(subject_boxes, reference_boxes, params):
    """
    If subject count > 0, reference count must also be > 0.

    This implements 'X requires Y'. Parameters:
        _subject_count: passed by evaluator
        _reference_count: passed by evaluator
    """
    sc = params.get("_subject_count", 0)
    rc = params.get("_reference_count", 0)
    if sc == 0:
        return PredicateResult(
            passed=True,
            evaluable=True,
            detail="requires_component: subject absent, rule vacuously satisfied",
        )
    passed = rc > 0
    return PredicateResult(
        passed=passed,
        evaluable=True,
        detail=f"requires_component: subject_count={sc}, reference_count={rc}",
    )


def require_all_present(subject_boxes, reference_boxes, params):
    """
    Check that ALL components in 'component_counts' have non-zero counts.

    This is the 'critical presence' gate: if any listed component is missing,
    the predicate fails.

    Parameters:
        component_counts: dict of {component_name: count}, passed by evaluator
        components: list of component names that must all be present
    """
    required = params["components"]
    counts = params.get("_all_counts", {})
    missing = [c for c in required if counts.get(c, 0) == 0]
    passed = len(missing) == 0
    return PredicateResult(
        passed=passed,
        evaluable=True,
        detail=(
            f"require_all_present: {required} -> "
            + ("all present" if passed else f"missing {missing}")
        ),
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

PREDICATE_REGISTRY: Dict[str, Callable] = {
    # Counts
    "count_exact": count_exact,
    "count_in_range": count_in_range,

    # Extremum
    "at_extremum": at_extremum,

    # Spatial containment
    "containment": containment,
    "forbid_containment": forbid_containment,

    # Positional
    "relative_position": relative_position,
    "below_centroid": below_centroid,
    "above_or_lateral_centroid": above_or_lateral_centroid,
    "require_proximity": require_proximity,

    # Size
    "size_ratio_in_range": size_ratio_in_range,
    "size_hierarchy": size_hierarchy,

    # Relational / set logic
    "requires_component": requires_component,
    "require_all_present": require_all_present,
}


def get_predicate(name: str) -> Callable:
    """Look up a predicate by name. Raises KeyError if unknown."""
    if name not in PREDICATE_REGISTRY:
        raise KeyError(
            f"Unknown predicate '{name}'. "
            f"Available: {sorted(PREDICATE_REGISTRY.keys())}"
        )
    return PREDICATE_REGISTRY[name]


def list_predicates() -> List[str]:
    """Return the names of all registered predicates."""
    return sorted(PREDICATE_REGISTRY.keys())