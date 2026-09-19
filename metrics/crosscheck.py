#!/usr/bin/env python3
"""
Baseline cross-check metrics for one domain: CLIPScore, SigLIP, SigLIP-2.

Same scoring as metrics/other_metrics_crosscheck.ipynb, but paths, caption
columns and device come from the domain config instead of being hard-coded to
one HPC car folder. Outputs keep the notebook's filenames and JSON/CSV schema,
so new runs stay comparable with results/animal/crosscheck/ from May 2026.

    python metrics/crosscheck.py --config configs/car.yaml --metric clipscore
    python metrics/crosscheck.py --config configs/car.yaml --metric siglip2 --device cuda:1
    python metrics/crosscheck.py --config configs/animal.yaml --metric all

Writes into results/<domain>/crosscheck/:
    clipscore_results_<domain>.csv     per image
    clipscore_summary_<domain>.json    mean/std/min/max/median
    siglip2_scores_<domain>.json       per image + summary + failures
    siglip2_summary_<domain>.json      summary only
    (siglip_*.json for the SigLIP-1 variant)

VQAScore and FLEUR are not here: VQAScore runs in its own project/environment
(t2v-metrics), FLEUR from metrics/fleur.py, both needing their own installs.
Feed every metric to metrics/crosscheck_summary.py for the Table IV rows.
"""

import argparse
import json
import statistics as st
import sys
from pathlib import Path

import pandas as pd
import torch
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.config_manager import ConfigManager                      # noqa: E402

SIGLIP_CHECKPOINTS = {"siglip": "google/siglip-base-patch16-224",
                      "siglip2": "google/siglip2-so400m-patch14-384"}


def load_pairs(config: ConfigManager, images_dir: Path):
    """[(image filename, caption)] for every image that exists on disk."""
    csv_path = config.require("paths.captions_csv")
    image_col = config.get("csv_mapping.image_col", "Image_path")
    caption_col = config.get("csv_mapping.caption_col", "Image_caption")
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    rows, missing = [], []
    for _, row in df.iterrows():
        name = str(row[image_col]).strip().split("-> ")[-1].strip()
        name = Path(name).name
        if not name.lower().endswith((".png", ".jpg", ".jpeg")):
            name += ".png"
        (rows if (images_dir / name).exists() else missing).append(
            (name, str(row[caption_col]).strip()))
    if missing:
        print(f"WARNING: {len(missing)} caption rows have no image "
              f"(first: {[m[0] for m in missing[:3]]})")
    print(f"{len(rows)} image-caption pairs from {csv_path}")
    return rows, missing


def summarize(scores, total, failed, key):
    return {
        "total_images": total,
        "successful_evaluations": len(scores),
        "failed_evaluations": failed,
        f"mean_{key}": float(st.mean(scores)) if scores else 0.0,
        f"std_{key}": float(st.pstdev(scores)) if scores else 0.0,
        f"min_{key}": float(min(scores)) if scores else 0.0,
        f"max_{key}": float(max(scores)) if scores else 0.0,
        f"median_{key}": float(st.median(scores)) if scores else 0.0,
    }


def run_clipscore(pairs, images_dir, out_dir, domain, device, config):
    """OpenAI CLIP ViT-B/32 cosine similarity x 100, as in the notebook."""
    import clip

    model, preprocess = clip.load("ViT-B/32", device=device)
    image_col = config.get("csv_mapping.image_col", "Image_path")
    caption_col = config.get("csv_mapping.caption_col", "Image_caption")

    records, scores, failed = [], [], 0
    for i, (name, caption) in enumerate(pairs, 1):
        try:
            with Image.open(images_dir / name) as im:
                image = preprocess(im.convert("RGB")).unsqueeze(0).to(device)
            text = clip.tokenize([caption], truncate=True).to(device)
            with torch.no_grad():
                f_i = model.encode_image(image)
                f_t = model.encode_text(text)
                f_i = f_i / f_i.norm(dim=-1, keepdim=True)
                f_t = f_t / f_t.norm(dim=-1, keepdim=True)
                score = (f_i @ f_t.T).item() * 100
        except Exception as exc:                                   # noqa: BLE001
            print(f"  error on {name}: {exc}")
            score, failed = None, failed + 1
        records.append({caption_col: caption, image_col: name,
                        "image_filename": name, "CLIPScore": score})
        if score is not None:
            scores.append(score)
        if i % 10 == 0:
            print(f"  {i}/{len(pairs)}")

    pd.DataFrame(records).to_csv(out_dir / f"clipscore_results_{domain}.csv", index=False)
    summary = summarize(scores, len(pairs), failed, "clipscore")
    (out_dir / f"clipscore_summary_{domain}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def as_tensor(features):
    """get_image_features/get_text_features return a tensor on transformers 4.x
    and a BaseModelOutputWithPooling on 5.x. Normalise to the embedding tensor."""
    if isinstance(features, torch.Tensor):
        return features
    pooled = getattr(features, "pooler_output", None)
    if pooled is not None:
        return pooled
    hidden = getattr(features, "last_hidden_state", None)
    if hidden is not None:
        return hidden.mean(dim=1)
    raise TypeError(f"cannot read embeddings from {type(features).__name__}")


def run_siglip(pairs, images_dir, out_dir, domain, device, metric):
    """SigLIP / SigLIP-2 sigmoid score, as in the notebook.

    Note the sigmoid of a temperature-scaled cosine saturates near 1.0 for any
    roughly-matching caption - that near-zero spread is exactly what the paper
    reports for SigLIP-2. `siglip_similarity` keeps the unsquashed value.
    """
    from transformers import AutoImageProcessor, AutoModel, AutoTokenizer
    import torch.nn.functional as F

    checkpoint = SIGLIP_CHECKPOINTS[metric]
    print(f"Loading {checkpoint}")
    image_processor = AutoImageProcessor.from_pretrained(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModel.from_pretrained(checkpoint).to(device).eval()

    results, failures, scores = [], [], []
    for i, (name, caption) in enumerate(pairs, 1):
        try:
            image = Image.open(images_dir / name).convert("RGB")
            image_inputs = image_processor(images=image, return_tensors="pt").to(device)
            text_inputs = tokenizer(caption, return_tensors="pt", padding="max_length",
                                    truncation=True, max_length=64).to(device)
            with torch.no_grad():
                emb_i = F.normalize(as_tensor(model.get_image_features(**image_inputs)), p=2, dim=-1)
                emb_t = F.normalize(as_tensor(model.get_text_features(**text_inputs)), p=2, dim=-1)
                logits = torch.matmul(emb_i, emb_t.t())
                if hasattr(model, "logit_scale"):
                    logits = logits * model.logit_scale.exp()
                sigmoid = float(torch.sigmoid(logits)[0][0].item())
                similarity = float(logits[0][0].item())
        except Exception as exc:                                   # noqa: BLE001
            print(f"  error on {name}: {exc}")
            failures.append({"image_path": name, "given_caption": caption, "error": str(exc)})
            continue
        results.append({"image_path": name, "given_caption": caption,
                        "siglip_score": sigmoid, "siglip_similarity": similarity})
        scores.append(sigmoid)
        if i % 10 == 0:
            print(f"  {i}/{len(pairs)}")

    summary = summarize(scores, len(pairs), len(failures), "siglip_score")
    (out_dir / f"{metric}_scores_{domain}.json").write_text(json.dumps(
        {"summary": summary, "detailed_results": results,
         "failed_evaluations": failures}, indent=2), encoding="utf-8")
    (out_dir / f"{metric}_summary_{domain}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="Domain config YAML")
    ap.add_argument("--metric", default="all", choices=["clipscore", "siglip", "siglip2", "all"])
    ap.add_argument("--images", help="Image folder (default: data/<domain>/images)")
    ap.add_argument("--out-dir", help="Default: results/<domain>/crosscheck")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    config = ConfigManager.load(args.config)
    rules = json.loads(Path(config.require("paths.rules_file")).read_text(encoding="utf-8"))
    domain = rules.get("object_type", "domain")
    images_dir = Path(args.images or f"data/{domain}/images")
    out_dir = Path(args.out_dir or f"results/{domain}/crosscheck")
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs, _ = load_pairs(config, images_dir)
    if not pairs:
        sys.exit(f"No images found in {images_dir}")

    wanted = ["clipscore", "siglip2"] if args.metric == "all" else [args.metric]
    for metric in wanted:
        print(f"\n===== {domain}: {metric} on {args.device} =====")
        if metric == "clipscore":
            summary = run_clipscore(pairs, images_dir, out_dir, domain, args.device, config)
        else:
            summary = run_siglip(pairs, images_dir, out_dir, domain, args.device, metric)
        print(json.dumps(summary, indent=2))
    print(f"\nWrote {out_dir}")


if __name__ == "__main__":
    main()
