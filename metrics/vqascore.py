#!/usr/bin/env python3
"""
VQAScore (clip-flant5-xxl) for one domain, as a PCMDE baseline cross-check.

Scores every image against its own caption from the domain's captions CSV and
writes results/<domain>/crosscheck/vqascore_<domain>.json in the schema
metrics/crosscheck_summary.py reads.

Run it in whichever environment has t2v-metrics, from any directory: it chdirs
to the PCMDE repo, so images come from and results land in PCMDE regardless of
where you launch it.

    # from the VQAScore project, in its own environment
    python /home/kgupta/workspace/Synthetic_Data_gen/pcmde/metrics/vqascore.py \
           --config configs/car.yaml
    python .../pcmde/metrics/vqascore.py --config configs/animal.yaml --limit 3

Unlike the baseline-vs-finetuned comparison script this came from, captions are
matched to images by filename through the config's csv_mapping, not by position
in a prompts file, so a missing or reordered row cannot silently shift every
score onto the wrong image.
"""

import argparse
import json
import os
import statistics as st
import sys
from pathlib import Path

# t2v_metrics probes flash-attention at import time and fails on some boxes.
os.environ.setdefault("FLASH_ATTENTION_SKIP_CUDA_CHECK", "1")

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.config_manager import ConfigManager                      # noqa: E402


def load_pairs(config: ConfigManager, images_dir: Path):
    """[(filename, caption)] for every caption row whose image exists."""
    image_col = config.get("csv_mapping.image_col", "Image_path")
    caption_col = config.get("csv_mapping.caption_col", "Image_caption")
    df = pd.read_csv(config.require("paths.captions_csv"))
    df.columns = [c.strip() for c in df.columns]

    pairs, missing = [], []
    for _, row in df.iterrows():
        name = Path(str(row[image_col]).strip().split("-> ")[-1].strip()).name
        if not name.lower().endswith((".png", ".jpg", ".jpeg")):
            name += ".png"
        (pairs if (images_dir / name).exists() else missing).append(
            (name, str(row[caption_col]).strip()))
    if missing:
        print(f"Skipping {len(missing)} caption row(s) with no image: "
              f"{[m[0] for m in missing[:5]]}")
    return pairs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="Domain config YAML")
    ap.add_argument("--images", help="Image folder (default: data/<domain>/images)")
    ap.add_argument("--out-dir", help="Default: results/<domain>/crosscheck")
    ap.add_argument("--model", default="clip-flant5-xxl",
                    help="t2v_metrics VQAScore model (clip-flant5-xl is smaller)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, help="Score only the first N images")
    args = ap.parse_args()

    # This script is normally launched from the VQAScore project (its own env),
    # while every path in the config is relative to the PCMDE repo. Resolve the
    # config against the launch directory first, then work from the repo root so
    # images are read from and results written to PCMDE, not the launch folder.
    config_path = Path(args.config)
    if not config_path.is_absolute() and not config_path.exists():
        config_path = PROJECT_ROOT / config_path
    config_path = config_path.resolve()
    os.chdir(PROJECT_ROOT)
    print(f"Working directory: {PROJECT_ROOT}")

    config = ConfigManager.load(str(config_path))
    rules = json.loads(Path(config.require("paths.rules_file")).read_text(encoding="utf-8"))
    domain = rules.get("object_type", "domain")
    images_dir = Path(args.images or f"data/{domain}/images")
    out_dir = Path(args.out_dir or f"results/{domain}/crosscheck")
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = load_pairs(config, images_dir)
    if args.limit:
        pairs = pairs[:args.limit]
    if not pairs:
        sys.exit(f"No image-caption pairs found for {domain} in {images_dir}")
    print(f"{domain}: {len(pairs)} image-caption pairs")

    import t2v_metrics
    print(f"Loading VQAScore model {args.model} ...")
    scorer = t2v_metrics.VQAScore(model=args.model, device=args.device)

    results, failures, scores = [], [], []
    for i, (name, caption) in enumerate(pairs, 1):
        try:
            raw = scorer(images=[str(images_dir / name)], texts=[caption])
            score = float(raw.reshape(-1)[0])          # (1 image, 1 text) -> scalar
        except Exception as exc:                                   # noqa: BLE001
            print(f"  error on {name}: {exc}")
            failures.append({"image_path": name, "given_caption": caption, "error": str(exc)})
            continue
        results.append({"image_path": name, "given_caption": caption, "vqascore": score})
        scores.append(score)
        if i % 10 == 0 or i == len(pairs):
            print(f"  {i}/{len(pairs)}  last={score:.4f}")

    summary = {
        "model": args.model,
        "total_images": len(pairs),
        "successful_evaluations": len(scores),
        "failed_evaluations": len(failures),
        "mean_vqascore": float(st.mean(scores)) if scores else 0.0,
        "std_vqascore": float(st.pstdev(scores)) if scores else 0.0,
        "min_vqascore": float(min(scores)) if scores else 0.0,
        "max_vqascore": float(max(scores)) if scores else 0.0,
        "median_vqascore": float(st.median(scores)) if scores else 0.0,
    }
    (out_dir / f"vqascore_{domain}.json").write_text(json.dumps(
        {"summary": summary, "detailed_results": results,
         "failed_evaluations": failures}, indent=2), encoding="utf-8")
    pd.DataFrame(results).to_csv(out_dir / f"vqascore_scores_{domain}.csv", index=False)
    print(json.dumps(summary, indent=2))
    print(f"Wrote {out_dir / f'vqascore_{domain}.json'}")


if __name__ == "__main__":
    main()
