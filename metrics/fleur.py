# my_eval.py — FLEUR evaluation for your custom image-prompt dataset

import os, re, time, json, torch
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import TextStreamer

import sys
sys.path.insert(0, './LLaVA')

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria


MODEL_PATH = "liuhaotian/llava-v1.5-13b"


def parse_args():
    """Paths come from the domain config; --csv/--images/--out-dir override them."""
    import argparse
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from core.config_manager import ConfigManager

    ap = argparse.ArgumentParser(description="FLEUR (LLaVA-1.5-13B) caption scoring")
    ap.add_argument("--config", help="Domain config YAML, e.g. configs/car.yaml")
    ap.add_argument("--csv", help="Captions CSV (default: paths.captions_csv)")
    ap.add_argument("--images", help="Image folder (default: data/<domain>/images)")
    ap.add_argument("--out-dir", help="Default: results/<domain>/crosscheck")
    ap.add_argument("--model", default=MODEL_PATH)
    ap.add_argument("--device-map", default="cuda:0",
                    help="Single device (default) or 'auto' to shard across GPUs")
    ap.add_argument("--limit", type=int, help="Score only the first N images (smoke test)")
    a = ap.parse_args()

    domain, image_col, caption_col, csv_path = "custom", "Image_path", "Image_caption", a.csv
    if a.config:
        cfg = ConfigManager.load(a.config)
        rules = json.loads(Path(cfg.require("paths.rules_file")).read_text(encoding="utf-8"))
        domain = rules.get("object_type", "domain")
        image_col = cfg.get("csv_mapping.image_col", image_col)
        caption_col = cfg.get("csv_mapping.caption_col", caption_col)
        csv_path = csv_path or cfg.require("paths.captions_csv")
    if not csv_path:
        ap.error("pass --config or --csv")
    return {
        "domain": domain, "csv": csv_path, "model": a.model,
        "images": a.images or f"data/{domain}/images",
        "out_dir": a.out_dir or f"results/{domain}/crosscheck",
        "image_col": image_col, "caption_col": caption_col,
        "device_map": a.device_map, "limit": a.limit,
    }


def run_fleur():
    opts = parse_args()
    CSV_PATH, IMAGE_DIR, MODEL_PATH = opts["csv"], opts["images"], opts["model"]
    OUT_DIR = opts["out_dir"]

    df = pd.read_csv(CSV_PATH)
    df.columns = [c.strip() for c in df.columns]
    # Caption CSVs can carry rows whose image does not exist (aircraft has one),
    # which would otherwise land in the output as a null score.
    present = df[opts["image_col"]].apply(
        lambda p: os.path.exists(os.path.join(IMAGE_DIR, os.path.basename(str(p).strip()))))
    if not present.all():
        print(f"Skipping {(~present).sum()} caption row(s) with no image: "
              f"{df.loc[~present, opts['image_col']].tolist()[:5]}")
        df = df[present].reset_index(drop=True)
    if opts["limit"]:
        df = df.head(opts["limit"]).reset_index(drop=True)
    img_paths = df[opts["image_col"]].tolist()
    captions = df[opts["caption_col"]].tolist()
    print(f"FLEUR: {len(df)} pairs from {CSV_PATH}, images in {IMAGE_DIR}")

    # Load LLaVA model. 13B in fp16 is ~26 GB and fits one 48 GB card, so keep it
    # on a single device: device_map="auto" adds accelerate hooks that a later
    # .to() breaks, and the builder already loads in fp16.
    print(f"Loading LLaVA model on {opts['device_map']} ...")
    model_name = get_model_name_from_path(MODEL_PATH)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=MODEL_PATH,
        model_base=None,
        model_name=model_name,
        device_map=opts["device_map"],
    )

    rate2token = {s: tokenizer.encode(str(s))[-1] for s in range(10)}
    conv_mode  = "llava_v1"

    # Set up results file
    os.makedirs(OUT_DIR, exist_ok=True)
    timestamp   = time.strftime('%y%m%d%H%M%S', time.localtime())
    result_path = f"{OUT_DIR}/fleur_log_{opts['domain']}_{timestamp}.txt"
    result_file = open(result_path, 'w')
    result_file.write("FLEUR Evaluation — Custom Dataset\n")
    result_file.write("=" * 60 + "\n\n")

    all_scores = []

    for idx, (img_file, caption) in enumerate(zip(tqdm(img_paths), captions)):
        result_file.write(f"[{idx+1}] Image: {img_file}\n")
        result_file.write(f"Caption: {caption[:120]}...\n")  # truncate for readability

        full_img_path = os.path.join(IMAGE_DIR, img_file)
        if not os.path.exists(full_img_path):
            print(f"  WARNING: Image not found — {full_img_path}")
            result_file.write("  ERROR: Image not found\n\n")
            all_scores.append(None)
            continue

        # Load and preprocess image
        image = Image.open(full_img_path).convert('RGB')

        class FakeArgs:
            image_aspect_ratio = 'pad'
        image_tensor = process_images([image], image_processor, FakeArgs())
        if isinstance(image_tensor, list):
            image_tensor = [t.to(model.device, dtype=torch.float16) for t in image_tensor]
        else:
            image_tensor = image_tensor.to(model.device, dtype=torch.float16)

        # Build FLEUR prompt
        inp = (
            f"Your task is to evaluate and rate the caption on a scale of 0.0 to 1.0 "
            f"based on the given Grading Criteria. (Print Real Number Score ONLY)\n\n"
            f"Grading Criteria:\n\n"
            f"0.0: The caption does not describe the image at all.\n"
            f"1.0: The caption accurately and clearly describes the image.\n\n"
            f"Caption: {caption}\n\n"
            f"Score(Choose a rating from 0.0 to 1.0):"
        )

        conv = conv_templates[conv_mode].copy()
        if model.config.mm_use_im_start_end:
            inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
        else:
            inp = DEFAULT_IMAGE_TOKEN + '\n' + inp

        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt    = conv.get_prompt()
        input_ids = tokenizer_image_token(
            prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
        ).unsqueeze(0).to(model.device)

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

        with torch.inference_mode():
            output_dict = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=False,
                temperature=0.2,
                num_beams=1,
                max_new_tokens=64,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
                output_scores=True,
                return_dict_in_generate=True,
            )

        output_ids = output_dict.sequences
        # LLaVA v1.2.0 generates from inputs_embeds, so `sequences` holds ONLY the
        # new tokens; older versions returned prompt + completion. Slice only when
        # the prompt is actually there, or the reply is sliced away entirely.
        gen_ids = (output_ids[0, input_ids.shape[1]:]
                   if output_ids.shape[1] > input_ids.shape[1] else output_ids[0])
        outputs = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        # Score smoothing (FLEUR's core scoring method)
        score = None
        try:
            # The original chopped a trailing "</s>" by slicing [:-4]; depending on
            # the transformers version the decode has no such suffix, and slicing
            # then eats the score itself. Strip the stop string if it is there,
            # then take the first number in the reply.
            text = outputs
            for suffix in (stop_str, "</s>", "<|im_end|>"):
                if suffix and text.endswith(suffix):
                    text = text[: -len(suffix)].strip()
                    break
            match = re.search(r'\d*\.?\d+', text)
            if not match:
                raise ValueError(f"no number in reply: {outputs!r}")
            score_check     = float(match.group())

            if not (0.0 <= score_check <= 1.0):
                raise ValueError("Out of range")

            if score_check < 1.0:
                num_index_in_score = str(score_check).index('.') + 1
                find_num           = int(str(score_check)[num_index_in_score])
                num_index_in_token = (gen_ids == rate2token[find_num]).nonzero().squeeze()
                if len(num_index_in_token.shape) > 0:
                    num_index_in_token = num_index_in_token[1] if find_num == 0 else num_index_in_token[0]
                probs = torch.nn.functional.softmax(output_dict.scores[num_index_in_token], dim=-1)[0]
                score = sum(probs[token].item() * rate * 0.1 for rate, token in rate2token.items())

                if len(str(score_check)) > 3:
                    num2 = int(str(score_check)[str(score_check).index('.') + 2])
                    idx2 = (gen_ids == rate2token[num2]).nonzero().squeeze()
                    if len(idx2.shape) > 0:
                        idx2 = idx2[1]
                    probs2 = torch.nn.functional.softmax(output_dict.scores[idx2], dim=-1)[0]
                    score += sum(probs2[token].item() * rate * 0.01 for rate, token in rate2token.items())
            else:
                idx1  = (gen_ids == rate2token[1]).nonzero().squeeze()
                probs = torch.nn.functional.softmax(output_dict.scores[idx1], dim=-1)[0]
                score = 0.9 * probs[rate2token[0]].item() + probs[rate2token[1]].item()

        except Exception as e:
            print(f"  Scoring error for {img_file}: {e}")
            result_file.write(f"  SCORING ERROR: {e}\n\n")
            all_scores.append(None)
            continue

        print(f"  Raw output: {outputs!r}  |  Smoothed score: {score:.4f}")
        result_file.write(f"Raw model output : {outputs}\n")
        result_file.write(f"FLEUR score      : {score:.4f}\n\n")
        all_scores.append(score)

    # Summary
    valid = [s for s in all_scores if s is not None]
    if not valid:
        result_file.write("No image produced a parseable score.\n")
        result_file.close()
        raise SystemExit(f"FLEUR scored 0 of {len(all_scores)} images - see {result_path}")
    result_file.write("=" * 60 + "\n")
    result_file.write(f"Images evaluated : {len(valid)} / {len(all_scores)}\n")
    result_file.write(f"Mean FLEUR score : {np.mean(valid):.4f}\n")
    result_file.write(f"Median FLEUR score: {np.median(valid):.4f}\n")
    result_file.write(f"Std dev          : {np.std(valid):.4f}\n")
    result_file.write(f"Min score        : {np.min(valid):.4f}\n")
    result_file.write(f"Max score        : {np.max(valid):.4f}\n")
    result_file.close()

    print(f"\n✅ Done! Results saved to: {result_path}")
    print(f"Mean FLEUR score across {len(valid)} images: {np.mean(valid):.4f}")

    # Also save scores to CSV for easy analysis
    df["fleur_score"] = all_scores
    csv_out = f"{OUT_DIR}/fleur_scores_{opts['domain']}.csv"
    df.to_csv(csv_out, index=False)
    summary_out = f"{OUT_DIR}/fleur_summary_{opts['domain']}.json"
    with open(summary_out, "w") as fh:
        json.dump({"total_images": len(all_scores), "successful_evaluations": len(valid),
                   "mean_fleur": float(np.mean(valid)), "std_fleur": float(np.std(valid)),
                   "min_fleur": float(np.min(valid)), "max_fleur": float(np.max(valid)),
                   "median_fleur": float(np.median(valid))}, fh, indent=2)
    print(f"Per-image scores saved to: {csv_out}")
    print(f"Summary saved to: {summary_out}")

if __name__ == "__main__":
    run_fleur()