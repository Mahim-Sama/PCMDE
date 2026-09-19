"""
Stage 1a for the animal domain: Grounding DINO (Swin-B) + SAM + geometric rules.

RECONSTRUCTION. The script that produced the original 70-image
data/animal/detections/groundingdino_sam_animal.json was never committed
(docs/KNOWN_GAPS.md §3). This one follows the method in docs/DETECTION_SETUP.md
and reproduces that file's schema and conventions (one head at confidence 0.95,
one torso, per-species part inventory, off-inventory parts as `foreign_part`),
but its numbers will not match the old file box for box. Run it over all 100
images so the 70 old and 30 new images come from the same detector.

Method, per image:
  1. Species pass: prompt the eight species, keep the best-scoring animal box.
  2. Part pass: one prompt per species-specific part (plus off-inventory parts,
     which become `foreign_part`) at every --scales inference resolution.
  3. SAM turns every candidate box into a mask; duplicate suppression uses mask
     IoU, so touching or crossing limbs are not deleted.
  4. Geometric rules: head reconstruction, torso bounds, limb validity filter.

Uses the Hugging Face ports of Grounding DINO and SAM, so nothing has to be
compiled (works on Windows).

    python pipeline/detection/groundingdino_sam_animal.py
    python pipeline/detection/groundingdino_sam_animal.py --limit 5 --annotate results/animal/detections_annotated
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("USE_TF", "0")

import torch
from PIL import Image, ImageDraw
from transformers import (AutoModelForZeroShotObjectDetection, AutoProcessor,
                          SamModel, SamProcessor)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Order of the `components` block in the output, as in the original file.
ALL_CLASSES = ["head", "eye", "ear", "nose", "beak", "horn", "trunk",
               "leg", "tail", "wing", "torso", "foreign_part"]

# Text prompted for each class. Elephant "horns" are tusks.
PHRASES = {"head": "head", "eye": "eye", "ear": "ear", "nose": "nose", "beak": "beak",
           "horn": "horn", "trunk": "trunk", "leg": "leg", "tail": "tail",
           "wing": "wing", "torso": "torso"}
SPECIES_PHRASE_OVERRIDES = {"elephant": {"horn": "tusk"}}

MAMMAL = ["head", "eye", "ear", "nose", "leg", "tail", "torso"]
INVENTORY = {
    "dog": MAMMAL, "cat": MAMMAL, "horse": MAMMAL, "squirrel": MAMMAL,
    "cow": MAMMAL + ["horn"], "sheep": MAMMAL + ["horn"],
    "elephant": ["head", "eye", "ear", "trunk", "horn", "leg", "tail", "torso"],
    "chicken": ["head", "eye", "beak", "leg", "tail", "wing", "torso"],
}
# Parts prompted for on every image; any outside the species' inventory become
# foreign_part. Ear and nose are not candidates: on chickens GDINO puts those
# labels on combs and wattles far more often than on a real anomaly.
FOREIGN_CANDIDATES = ["beak", "wing", "horn", "trunk"]
SPECIES = list(INVENTORY)

FACIAL = {"eye", "ear", "nose", "beak", "horn", "trunk"}
# Features the head box is rebuilt from. Trunks and tusks reach the ground and
# would stretch the head over the whole animal.
HEAD_FEATURES = ["eye", "ear", "nose", "beak"]
LIMBS = {"leg", "tail", "wing"}
SMALL_PARTS = {"eye", "nose", "beak"}   # GDINO scores these low even when right

HEAD_CONFIDENCE = 0.95      # reconstructed head, as in the original file
SYNTH_TORSO_CONFIDENCE = 0.90
HEAD_PADDING = 0.20         # per side, as a fraction of the feature union size
LIMB_AREA_RATIO = 1.2       # limb mask area <= 1.2 x torso box area
TORSO_CORE_PADDING = 0.15
MASK_IOU_SAME_CLASS = 0.5
SAME_CLASS_CONTAINMENT = 0.8
MASK_IOU_CROSS_CLASS = 0.7
FOREIGN_OVERLAP = 0.3


# ---------------------------------------------------------------------------
# Box helpers ([x1, y1, x2, y2] in pixels)
# ---------------------------------------------------------------------------

def area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def box_iou(a, b):
    inter = area([max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])])
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0.0


def box_containment(a, b):
    """Share of the smaller box that lies inside the larger one."""
    inter = area([max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])])
    smaller = min(area(a), area(b))
    return inter / smaller if smaller > 0 else 0.0


def intersects(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def pad(b, frac, w, h):
    px, py = (b[2] - b[0]) * frac, (b[3] - b[1]) * frac
    return clip([b[0] - px, b[1] - py, b[2] + px, b[3] + py], w, h)


def clip(b, w, h):
    return [min(max(b[0], 0.0), w), min(max(b[1], 0.0), h),
            min(max(b[2], 0.0), w), min(max(b[3], 0.0), h)]


def union(boxes):
    return [min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes)]


def center(b):
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class Detector:
    def __init__(self, gdino_id: str, sam_id: str, device: str):
        self.device = device
        print(f"Loading {gdino_id} and {sam_id} on {device} ...")
        self.gdino_proc = AutoProcessor.from_pretrained(gdino_id)
        self.gdino = AutoModelForZeroShotObjectDetection.from_pretrained(gdino_id).to(device).eval()
        self.sam_proc = SamProcessor.from_pretrained(sam_id)
        self.sam = SamModel.from_pretrained(sam_id).to(device).eval()

    @torch.no_grad()
    def ground(self, image: Image.Image, classes: Dict[str, str], scale: float,
               top_k: int = 30) -> List[Dict]:
        """Candidate boxes for `classes` ({class: phrase}).

        Each GDINO query is assigned to the class whose phrase tokens it scores
        highest on, which avoids HF's merged labels such as "eye ear".
        """
        names = list(classes)
        prompt, spans = "", []
        for name in names:
            phrase = classes[name].lower()
            spans.append((len(prompt), len(prompt) + len(phrase)))
            prompt += phrase + " . "
        prompt = prompt.strip()

        text = self.gdino_proc.tokenizer(prompt, return_offsets_mapping=True, return_tensors="pt")
        offsets = text.pop("offset_mapping")[0].tolist()
        token_idx = [[i for i, (s, e) in enumerate(offsets) if e > s and a <= s < b] for a, b in spans]

        size = {"shortest_edge": int(800 * scale), "longest_edge": int(1333 * scale)}
        pixels = self.gdino_proc.image_processor(image, size=size, return_tensors="pt")
        inputs = {**{k: v.to(self.device) for k, v in text.items()},
                  **{k: v.to(self.device) for k, v in pixels.items()}}
        out = self.gdino(**inputs)

        probs = out.logits[0].sigmoid()                       # (queries, tokens)
        per_class = torch.stack([probs[:, idx].max(dim=1).values for idx in token_idx], dim=1)
        scores, labels = per_class.max(dim=1)
        w, h = image.size
        cx, cy, bw, bh = out.pred_boxes[0].unbind(-1)
        boxes = torch.stack([(cx - bw / 2) * w, (cy - bh / 2) * h,
                             (cx + bw / 2) * w, (cy + bh / 2) * h], dim=-1).cpu().tolist()

        cands = []
        for ci, name in enumerate(names):
            sel = (labels == ci).nonzero().flatten().tolist()
            sel.sort(key=lambda q: -scores[q].item())
            kept = []
            for q in sel[:top_k * 3]:
                b = clip(boxes[q], w, h)
                # Box NMS only for near-identical queries; real suppression is mask-based.
                if area(b) > 1 and all(box_iou(b, k["bbox"]) < 0.9 for k in kept):
                    kept.append({"class": name, "confidence": scores[q].item(), "bbox": b})
                if len(kept) == top_k:
                    break
            cands += kept
        return cands

    @torch.no_grad()
    def masks(self, image: Image.Image, boxes: List[List[float]], chunk: int = 32) -> List[torch.Tensor]:
        if not boxes:
            return []
        inputs = self.sam_proc(image, return_tensors="pt")
        emb = self.sam.get_image_embeddings(inputs["pixel_values"].to(self.device))
        result = []
        for i in range(0, len(boxes), chunk):
            part = self.sam_proc(image, input_boxes=[boxes[i:i + chunk]], return_tensors="pt")
            out = self.sam(image_embeddings=emb, input_boxes=part["input_boxes"].to(self.device),
                           multimask_output=False)
            m = self.sam_proc.image_processor.post_process_masks(
                out.pred_masks.cpu(), part["original_sizes"], part["reshaped_input_sizes"])[0]
            result += [mk[0] for mk in m]
        return result


def mask_iou(a: torch.Tensor, b: torch.Tensor) -> float:
    inter = (a & b).sum().item()
    uni = (a | b).sum().item()
    return inter / uni if uni else 0.0


def mask_containment(a: Dict, b: Dict) -> float:
    """Share of the smaller mask that lies inside the larger one."""
    smaller = min(a["mask_area"], b["mask_area"])
    return (a["mask"] & b["mask"]).sum().item() / smaller if smaller else 0.0


# ---------------------------------------------------------------------------
# Per-image pipeline
# ---------------------------------------------------------------------------

def species_from_filename(name: str, rules: Dict) -> Optional[str]:
    stem = name.lower()
    for sid, spec in rules.get("subtypes", {}).items():
        for prefix in spec.get("filename_prefix_patterns", []):
            if re.match(rf"{re.escape(prefix.lower())}[_\-]?\d", stem):
                return sid
    return None


def detect_image(det: Detector, path: Path, rules: Dict, args) -> Dict:
    image = Image.open(path).convert("RGB")
    w, h = image.size

    # 1. Species
    sp_cands = det.ground(image, {s: s for s in SPECIES}, 1.0, top_k=3)
    sp_cands.sort(key=lambda c: -c["confidence"])
    by_name = species_from_filename(path.name, rules)
    by_model = sp_cands[0]["class"] if sp_cands else None
    species = by_name if args.species_source == "filename" and by_name else (by_model or by_name or "dog")
    if by_name and by_model and by_name != by_model:
        print(f"  ! {path.name}: detector says {by_model}, filename says {by_name}; using {species}")
    animal_boxes = [c["bbox"] for c in sp_cands if c["class"] == species]
    animal = animal_boxes[0] if animal_boxes else [0.0, 0.0, float(w), float(h)]
    region = pad(animal, 0.10, w, h)

    # 2. Parts at each scale
    inventory = INVENTORY.get(species, MAMMAL)
    phrases = {c: SPECIES_PHRASE_OVERRIDES.get(species, {}).get(c, PHRASES[c]) for c in inventory}
    foreign = [c for c in FOREIGN_CANDIDATES if c not in inventory]
    for c in foreign:
        phrases["foreign:" + c] = PHRASES[c]

    cands = []
    for scale in args.scales:
        # One prompt per class: in a joint 11-phrase prompt GDINO spreads its
        # score across phrases and clearly visible legs fall below threshold.
        grounded = [c for name, phrase in phrases.items()
                    for c in det.ground(image, {name: phrase}, scale)]
        for c in grounded:
            base = c["class"].split(":")[-1]
            thr = args.foreign_threshold if c["class"].startswith("foreign:") else \
                (args.small_threshold if base in SMALL_PARTS else args.threshold)
            if c["confidence"] < thr:
                continue
            cx, cy = center(c["bbox"])
            if not (region[0] <= cx <= region[2] and region[1] <= cy <= region[3]):
                continue                                     # background
            if base in FACIAL and area(c["bbox"]) > 0.25 * area(animal):
                continue                                     # whole-animal box labelled as a face part
            if base != "torso" and box_iou(c["bbox"], animal) > 0.6:
                continue                                     # whole-animal box labelled as a part
            cands.append(c)

    # 3. SAM masks, mask-based suppression
    masks = det.masks(image, [c["bbox"] for c in cands])
    for c, m in zip(cands, masks):
        c["mask"] = m
        c["mask_area"] = float(m.sum().item())
    cands.sort(key=lambda c: -c["confidence"])

    kept: List[Dict] = []
    for c in cands:
        dup = False
        for k in kept:
            iou = mask_iou(c["mask"], k["mask"])
            if c["class"] == k["class"] and (iou > MASK_IOU_SAME_CLASS or
                                             mask_containment(c, k) > SAME_CLASS_CONTAINMENT or
                                             box_containment(c["bbox"], k["bbox"]) > 0.9):
                dup = True                                   # includes an eye box nested in an eye box
            elif not c["class"].startswith("foreign:") and not k["class"].startswith("foreign:") \
                    and iou > MASK_IOU_CROSS_CLASS:
                dup = True                                   # e.g. an ear also called a horn
            if dup:
                break
        if not dup:
            kept.append(c)

    inventory_dets = [k for k in kept if not k["class"].startswith("foreign:")]
    foreign_dets = []
    for f in (k for k in kept if k["class"].startswith("foreign:")):
        # An off-inventory label on a region already explained by a real part is a
        # grounding confusion (ear read as horn), not an anomaly.
        explained = any(
            mask_iou(f["mask"], k["mask"]) > FOREIGN_OVERLAP or
            mask_containment(f, k) > 0.6
            for k in inventory_dets)
        if not explained:
            f["class"] = "foreign_part"
            foreign_dets.append(f)

    parts = {c: [k for k in inventory_dets if k["class"] == c] for c in inventory}

    # 4a. Head reconstruction: tight bound of facial features + 20% padding
    facial = [k["bbox"] for c in HEAD_FEATURES for k in parts.get(c, [])]
    gd_head = parts.get("head", [])
    head_box = None
    if facial:
        head_box = pad(union(facial), HEAD_PADDING, w, h)
        if gd_head and intersects(gd_head[0]["bbox"], head_box):
            head_box = union([head_box, gd_head[0]["bbox"]])
    elif gd_head:
        head_box = gd_head[0]["bbox"]
    head = [{"class": "head", "confidence": HEAD_CONFIDENCE, "bbox": head_box}] if head_box else []

    # 4b. Torso: below mid-head, above mid-legs
    mid_head = center(head_box)[1] if head_box else None
    legs = parts.get("leg", [])
    leg_ys = sorted(center(k["bbox"])[1] for k in legs)
    mid_legs = leg_ys[len(leg_ys) // 2] if leg_ys else None

    gd_torso = parts.get("torso", [])
    torso = None
    if gd_torso:
        t = list(gd_torso[0]["bbox"])
        top = max(t[1], mid_head) if mid_head is not None else t[1]
        bottom = min(t[3], mid_legs) if mid_legs is not None else t[3]
        if bottom - top > 0.1 * (t[3] - t[1]):
            t[1], t[3] = top, bottom
        torso = {"class": "torso", "confidence": gd_torso[0]["confidence"], "bbox": t}
    else:
        ax1, ay1, ax2, ay2 = animal
        if mid_head is not None and mid_legs is not None and mid_legs > mid_head:
            top, bottom = mid_head, mid_legs
        elif mid_head is not None:
            top = mid_head
            bottom = top + 0.5 * (ay2 - top)
        elif mid_legs is not None:
            bottom = mid_legs
            top = bottom - 0.5 * (bottom - ay1)
        else:
            top, bottom = ay1 + 0.25 * (ay2 - ay1), ay1 + 0.75 * (ay2 - ay1)
        torso = {"class": "torso", "confidence": SYNTH_TORSO_CONFIDENCE,
                 "bbox": clip([ax1, top, ax2, bottom], w, h)}

    # 4c. Limb validity filter
    core = pad(torso["bbox"], TORSO_CORE_PADDING, w, h)
    torso_area = area(torso["bbox"])
    for c in LIMBS & set(parts):
        parts[c] = [k for k in parts[c]
                    if k["mask_area"] <= LIMB_AREA_RATIO * torso_area and intersects(k["bbox"], core)]

    final = []
    for c in inventory:
        if c in ("head", "torso"):
            continue
        final += parts[c]
    final += head + [torso] + foreign_dets

    detections = [{"class": d["class"], "confidence": round(d["confidence"], 4),
                   "bbox": [round(v, 2) for v in d["bbox"]]} for d in final]
    components = {c: sum(1 for d in detections if d["class"] == c) for c in ALL_CLASSES}
    avg = sum(d["confidence"] for d in detections) / len(detections) if detections else 0.0

    if args.annotate:
        draw_annotations(image, detections, Path(args.annotate) / path.name)

    return {
        "image_id": path.name,
        "detected_animal": species,
        "components": components,
        "leg_count": components["leg"],
        "horn_count": components["horn"],
        "average_confidence": round(avg, 4),
        "total_detections": len(detections),
        "detections": detections,
    }


COLORS = {"head": "red", "torso": "blue", "leg": "lime", "tail": "orange", "eye": "cyan",
          "ear": "magenta", "nose": "yellow", "beak": "yellow", "horn": "white",
          "trunk": "pink", "wing": "purple", "foreign_part": "black"}


def draw_annotations(image: Image.Image, detections: List[Dict], out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    img = image.copy()
    d = ImageDraw.Draw(img)
    for det in detections:
        color = COLORS.get(det["class"], "white")
        d.rectangle(det["bbox"], outline=color, width=2)
        d.text((det["bbox"][0] + 2, det["bbox"][1] + 2), f'{det["class"]} {det["confidence"]:.2f}', fill=color)
    img.save(out)


# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Grounding DINO + SAM animal part detection (Stage 1a)")
    p.add_argument("--images", default="data/animal/images", help="Folder of images")
    p.add_argument("--output", default="data/animal/detections/groundingdino_sam_animal.json")
    p.add_argument("--rules", default="domains/animal/physical_rules_animal_v2.json",
                   help="Rules JSON; its filename_prefix_patterns cross-check the species")
    p.add_argument("--gdino-model", default="IDEA-Research/grounding-dino-base",
                   help="grounding-dino-base (Swin-B, as in the paper) or grounding-dino-tiny (Swin-T)")
    p.add_argument("--sam-model", default="facebook/sam-vit-base",
                   help="sam-vit-base fits a 4 GB GPU; use sam-vit-huge on the server")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--scales", type=float, nargs="+", default=[1.0, 1.5],
                   help="Inference resolutions relative to GDINO's 800 px default")
    p.add_argument("--threshold", type=float, default=0.25, help="Box threshold for most parts")
    p.add_argument("--small-threshold", type=float, default=0.15, help="Box threshold for eye/nose/beak")
    p.add_argument("--foreign-threshold", type=float, default=0.30, help="Box threshold for off-inventory parts")
    p.add_argument("--species-source", choices=["detector", "filename"], default="detector",
                   help="detector = two-step prompting as in the paper; filename = trust the cane_/gatto_ prefix")
    p.add_argument("--limit", type=int, help="Only the first N images (for a quick check)")
    p.add_argument("--annotate", help="Folder to write images with boxes drawn, for visual QA")
    p.add_argument("--force", action="store_true", help="Overwrite --output if it exists")
    args = p.parse_args()

    os.chdir(PROJECT_ROOT)
    out_path = Path(args.output)
    if out_path.exists() and not args.force and args.limit is None:
        sys.exit(f"{out_path} exists. Keep a copy (e.g. *.reference_70.json) and pass --force.")

    rules = json.loads(Path(args.rules).read_text(encoding="utf-8"))
    files = sorted(f for f in Path(args.images).iterdir()
                   if f.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if args.limit:
        files = files[:args.limit]
        if out_path.exists() and not args.force:
            out_path = out_path.with_name(out_path.stem + f".first{args.limit}.json")

    det = Detector(args.gdino_model, args.sam_model, args.device)
    results = []
    for i, f in enumerate(files, 1):
        r = detect_image(det, f, rules, args)
        results.append(r)
        print(f"[{i}/{len(files)}] {f.name}: {r['detected_animal']}, {r['total_detections']} boxes, "
              f"legs={r['leg_count']} foreign={r['components']['foreign_part']}")
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    payload = {
        "model": "GroundingDINO + SAM + Geometric Rules",
        "confidence_threshold": args.threshold,
        "settings": {"gdino_model": args.gdino_model, "sam_model": args.sam_model,
                     "scales": args.scales, "small_threshold": args.small_threshold,
                     "foreign_threshold": args.foreign_threshold,
                     "species_source": args.species_source,
                     "script": "pipeline/detection/groundingdino_sam_animal.py (reconstruction)"},
        "total_images": len(results),
        "total_detections": sum(r["total_detections"] for r in results),
        "results": results,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out_path} ({len(results)} images, {payload['total_detections']} boxes)")


if __name__ == "__main__":
    main()
