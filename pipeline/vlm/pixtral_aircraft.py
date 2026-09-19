import argparse
import os

# Paths and GPU are CLI options so the same script can run on any image set.
_parser = argparse.ArgumentParser(description="Pixtral-12B aircraft component counts")
_parser.add_argument("--images", default="data/aircraft/images",
                     help="Folder of images to run inference on")
_parser.add_argument("--output", default="data/aircraft/vlm/pixtral_aircraft.json",
                     help="Output JSON path")
_parser.add_argument("--device", default="0,1,2",
                     help="Value for CUDA_VISIBLE_DEVICES; the model is split across every listed GPU")
_parser.add_argument("--no-p2p", action="store_true",
                     help="Always copy tensors between GPUs through CPU memory (auto-enabled if the startup check fails)")
_args = _parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = _args.device
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Stop transformers from importing TensorFlow (installed but broken in this env).
os.environ["USE_TF"] = "0"

import traceback
from collections.abc import Mapping
import torch
import re
import json

import accelerate.hooks
from accelerate.utils import honor_type
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration


device = torch.device("cuda:0")

CONFIDENCE_THRESHOLD = 0.25


model_id = "mistral-community/pixtral-12b"

num_gpus = torch.cuda.device_count()


def gpu_copies_are_correct(num_gpus):
    # Some multi-GPU PCIe machines report direct GPU-to-GPU (P2P) access but the
    # copies return wrong data, which breaks a model split across GPUs.
    ok = True
    for src in range(num_gpus):
        x = torch.randint(0, 2**62, (8 * 1024 * 1024,), dtype=torch.int64, device=f"cuda:{src}")
        expected = x.cpu()
        for dst in range(num_gpus):
            if dst != src and not torch.equal(x.to(f"cuda:{dst}").cpu(), expected):
                print(f"GPU {src} -> GPU {dst}: direct copy returned wrong data")
                ok = False
    torch.cuda.empty_cache()
    return ok


def send_to_device_via_cpu(data, device, non_blocking=False, skip_keys=None):
    # Drop-in for accelerate's send_to_device: a tensor moving from one GPU to
    # another is copied through CPU memory instead of directly.
    if isinstance(skip_keys, str):
        skip_keys = [skip_keys]
    if isinstance(data, torch.Tensor):
        target = torch.device(f"cuda:{device}" if isinstance(device, int) else device)
        if data.is_cuda and target.type == "cuda" and data.device != target:
            data = data.cpu()
        return data.to(target)
    if isinstance(data, (tuple, list)):
        return honor_type(data, (send_to_device_via_cpu(d, device, skip_keys=skip_keys) for d in data))
    if isinstance(data, Mapping):
        return type(data)({
            k: v if skip_keys and k in skip_keys else send_to_device_via_cpu(v, device, skip_keys=skip_keys)
            for k, v in data.items()
        })
    return data


if num_gpus > 1:
    if _args.no_p2p or not gpu_copies_are_correct(num_gpus):
        print("Copying tensors between GPUs through CPU memory")
        # accelerate's device-map hooks move every tensor that crosses GPUs with this function.
        accelerate.hooks.send_to_device = send_to_device_via_cpu
    else:
        print("GPU-to-GPU copy check passed")

print(f"Splitting {model_id} across {num_gpus} GPU(s): CUDA_VISIBLE_DEVICES={_args.device}")

# ~25 GB of bfloat16 weights (the checkpoint's native dtype; float16 can overflow),
# so device_map="auto" spreads it over the visible GPUs. Inputs still go to cuda:0.
model = LlavaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
).eval()

processor = AutoProcessor.from_pretrained(model_id)
print("Model loaded!")


def move_inputs(value):
    # pixel_values must match the model dtype; depending on the transformers
    # version they arrive as a tensor or as nested lists of tensors.
    if isinstance(value, torch.Tensor):
        return value.to(device, dtype=model.dtype) if value.is_floating_point() else value.to(device)
    if isinstance(value, (list, tuple)):
        return type(value)(move_inputs(v) for v in value)
    return value


# -------------------------------
# Images (folder-based)
# -------------------------------
image_folder = _args.images
valid_extensions = [".png", ".jpg", ".jpeg", ".bmp"]

image_paths = [
    os.path.join(image_folder, f)
    for f in os.listdir(image_folder)
    if os.path.isfile(os.path.join(image_folder, f))
    and os.path.splitext(f)[1].lower() in valid_extensions
]

if not image_paths:
    raise FileNotFoundError(f"No valid images found in folder: {image_folder}")

# -------------------------------
# Regex for counting components
# -------------------------------
pattern = re.compile(
    r"(head|swept[\s\-_]?wing[s]?|tail[\s\-_]?wing[s]?|tail|engine[s]?)\D*([0-9]+|one|two|three|four|five|six|seven|eight|nine|zero)",
    re.IGNORECASE,
)

number_map = {
    "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "zero": 0
}

component_map = ["head", "tail", "swept_wing", "engine", "tail_wing"]

# -------------------------------
# Output JSON
# -------------------------------
output_json = {
    "detector": "Pixtral-12B",
    "total_images": len(image_paths),
    "results": []
}

# -------------------------------
# Prompt Template
prompt_template = (
    "[IMG] You are an aircraft component detector. Carefully examine this image and count ONLY the components that are clearly visible.\n\n"
    "Count these 5 parts:\n"
    "(a) Head - the nose/front fuselage (usually 1)\n"
    "(b) Tail - the rear fuselage section (usually 1)\n"
    "(c) Swept wings - main wings angled backward from the fuselage.\n"
    "   - Count ONLY the wings you can clearly see.\n"
    "   - If only one side wing (left OR right) is visible, count 1.\n"
    "   - If both wings (left AND right) are clearly visible, count 2.\n"
    "   - Do NOT assume symmetry — if one wing is hidden or unclear, count 1 or 0 accordingly.\n"
    "(d) Engines - visible jet engines or propellers (count each visible one)\n"
    "(e) Tail wings - horizontal stabilizers at the tail (count visible ones)\n\n"
    "IMPORTANT RULES:\n"
    "- Only count parts that are clearly visible in the image.\n"
    "- If you cannot see a component clearly, count it as 0.\n"
    "- Never assume hidden or occluded parts exist.\n"
    "- Use your best visual judgment and base all counts on visible evidence only.\n\n"
    "Respond EXACTLY in this format:\n"
    "(a) Head: [number]\n"
    "(b) Tail: [number]\n"
    "(c) Swept wings: [number]\n"
    "(d) Engines: [number]\n"
    "(e) Tail wings: [number]\n"
    "Overall confidence: [0.0–1.0]"
)



# -------------------------------
# Inference Loop
# -------------------------------
failures = 0

for idx, img_path in enumerate(image_paths, start=1):
    print(f"\nProcessing image {idx}/{len(image_paths)}: {img_path}")

    try:
        image = Image.open(img_path).convert("RGB")

        # Prepare input
        inputs = processor(text=prompt_template, images=image, return_tensors="pt")
        inputs = {k: move_inputs(v) for k, v in inputs.items()}

        # Generate output
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=512)
        # generate() returns prompt + answer; decode only the answer. The prompt's
        # "(usually 1)" / "visible one" lines otherwise match the regex below.
        raw_output = processor.tokenizer.batch_decode(
            output_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )[0]
    except Exception as e:
        print(f"Error {img_path}: {e}")
        # Full traceback once, so the failing library is visible.
        if failures == 0:
            traceback.print_exc()
        failures += 1
        # A CUDA assert leaves the GPUs unusable for the rest of this process.
        if "CUDA error" in str(e):
            raise SystemExit("Stopping: CUDA error, every later image would fail the same way.")
        continue

    print(f"Raw output:\n{raw_output}")

    # -------------------------------
    # Parse counts
    # -------------------------------
    components = {c: 0 for c in component_map}

    for line in raw_output.split("\n"):
        for part, count in pattern.findall(line):
            part_lower = part.lower().replace("-", "_").replace(" ", "_")
            try:
                count_val = int(number_map.get(count.lower(), count))
            except ValueError:
                continue

            if "tail_wing" in part_lower:
                key = "tail_wing"
            elif "swept_wing" in part_lower:
                key = "swept_wing"
            elif "engine" in part_lower:
                key = "engine"
            elif "head" in part_lower:
                key = "head"
            elif part_lower == "tail":
                key = "tail"
            else:
                continue

            components[key] = count_val

    # Confidence = fraction of detected components
    confidence = round(sum(1 for v in components.values() if v > 0) / len(component_map), 2)

    # Append result
    output_json["results"].append({
        "image_id": os.path.basename(img_path),
        "components": components,
        "confidence": confidence
    })

# -------------------------------
# Display and Save JSON
# -------------------------------
print("\n===== Final JSON Output =====")
print(json.dumps(output_json, indent=2, ensure_ascii=False))

output_file = _args.output

# Don't overwrite an existing results file when every image failed.
if not output_json["results"]:
    raise SystemExit(f"All {len(image_paths)} images failed; not writing {output_file}")

os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(output_json, f, indent=2, ensure_ascii=False)

print(f"\n JSON saved to: {output_file}")
