#!/usr/bin/env python3
"""
Check that a new detector run reproduces a reference detection file.

Use it before trusting re-downloaded weights: run inference on the images the
reference covers, then compare. Same weights + same thresholds should give the
same classes, the same per-image counts and near-identical boxes; small
confidence drift can come from a different ultralytics/torch/GPU.

Usage:
    python analysis/compare_detections.py \
        --reference data/car/detections/yolov12_car.reference_70.json \
        --new       data/car/detections/yolov12_car.json

Exit code 0 = reproduces the reference, 1 = it does not.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def load(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data, {r["image_id"]: r for r in data["results"]}


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match_boxes(ref_dets, new_dets, min_iou):
    """Greedy same-class matching. Returns (matched pairs, unmatched ref, unmatched new)."""
    pairs, used = [], set()
    for r in sorted(ref_dets, key=lambda d: -d["confidence"]):
        best, best_iou = None, min_iou
        for j, n in enumerate(new_dets):
            if j in used or n["class"] != r["class"]:
                continue
            value = iou(r["bbox"], n["bbox"])
            if value >= best_iou:
                best, best_iou = j, value
        if best is not None:
            used.add(best)
            pairs.append((r, new_dets[best], best_iou))
    return pairs, len(ref_dets) - len(pairs), len(new_dets) - len(used)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", required=True, help="Detection JSON known to be correct")
    ap.add_argument("--new", required=True, help="Detection JSON from the run being checked")
    ap.add_argument("--min-iou", type=float, default=0.9,
                    help="IoU for two same-class boxes to count as the same detection")
    ap.add_argument("--pass-rate", type=float, default=0.95,
                    help="Fraction of reference boxes that must be matched")
    args = ap.parse_args(argv)

    for label, path in (("--reference", args.reference), ("--new", args.new)):
        if not Path(path).is_file():
            print(f"Error: {label} file not found: {Path(path).resolve()}")
            if label == "--reference":
                print("  The reference is the ORIGINAL 70-image detection file. Copy it before "
                      "the inference cell overwrites data/<domain>/detections/yolov12_<domain>.json, "
                      "or restore it from a copy of the repo that still has it.")
            print("  Paths are relative to the directory you run from (the project root).")
            return 2

    ref_meta, ref = load(args.reference)
    new_meta, new = load(args.new)
    if len(ref) == len(new) and ref_meta.get("total_detections") == new_meta.get("total_detections") \
            and Path(args.reference).read_bytes() == Path(args.new).read_bytes():
        print("NOTE: the two files are byte-identical - the new run has not been written yet, "
              "or the reference was copied after the inference cell overwrote the original.")

    shared = sorted(set(ref) & set(new))
    only_ref = sorted(set(ref) - set(new))
    print(f"reference: {args.reference}  ({len(ref)} images)")
    print(f"new      : {args.new}  ({len(new)} images)")
    for key in ("confidence_threshold", "iou_threshold"):
        if ref_meta.get(key) != new_meta.get(key):
            print(f"  NOTE {key}: reference={ref_meta.get(key)} new={new_meta.get(key)}")
    print(f"images compared: {len(shared)}")
    if only_ref:
        print(f"  {len(only_ref)} reference image(s) missing from the new run: "
              f"{only_ref[:10]}{' ...' if len(only_ref) > 10 else ''}")
    if not shared:
        print("RESULT: nothing to compare - the two files share no image_id.")
        return 1

    ref_classes = Counter(d["class"] for i in shared for d in ref[i].get("detections", []))
    new_classes = Counter(d["class"] for i in shared for d in new[i].get("detections", []))
    if set(ref_classes) != set(new_classes):
        print(f"  CLASS NAMES DIFFER\n    reference: {sorted(ref_classes)}\n    new      : {sorted(new_classes)}")

    same_counts = matched = ref_total = new_total = 0
    conf_diffs, differing = [], []
    for image_id in shared:
        rd, nd = ref[image_id].get("detections", []), new[image_id].get("detections", [])
        if Counter(d["class"] for d in rd) == Counter(d["class"] for d in nd):
            same_counts += 1
        else:
            differing.append(image_id)
        pairs, _, _ = match_boxes(rd, nd, args.min_iou)
        matched += len(pairs)
        ref_total += len(rd)
        new_total += len(nd)
        conf_diffs += [abs(r["confidence"] - n["confidence"]) for r, n, _ in pairs]

    box_rate = matched / ref_total if ref_total else 1.0
    print(f"\nper-image class counts identical : {same_counts}/{len(shared)}")
    print(f"reference boxes matched (IoU>={args.min_iou}): {matched}/{ref_total} ({100 * box_rate:.1f}%)")
    print(f"boxes in new run                 : {new_total}")
    if conf_diffs:
        print(f"confidence |diff| on matched boxes: mean {sum(conf_diffs) / len(conf_diffs):.4f}, "
              f"max {max(conf_diffs):.4f}")
    if differing:
        print(f"images whose counts differ: {differing[:15]}{' ...' if len(differing) > 15 else ''}")

    ok = (set(ref_classes) == set(new_classes) and box_rate >= args.pass_rate
          and abs(new_total - ref_total) <= max(2, 0.05 * ref_total))
    print("\nRESULT: " + ("REPRODUCES the reference - these weights and settings match."
                          if ok else
                          "DOES NOT reproduce the reference - wrong weights, thresholds, "
                          "image size or library version. Do not use this run."))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
