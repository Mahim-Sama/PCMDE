import argparse
import os

# Paths and GPU are CLI options so the same script can run on any image set.
_parser = argparse.ArgumentParser(description="DeepSeek-VL2 animal component counts")
_parser.add_argument("--images", default="data/animal/images",
                     help="Folder of images to run inference on")
_parser.add_argument("--output", default="data/animal/vlm/deepseek_animal.json",
                     help="Output JSON path")
_parser.add_argument("--device", default="0,1,2",
                     help="Value for CUDA_VISIBLE_DEVICES; the model is split across every listed GPU")
_parser.add_argument("--model", default="deepseek-ai/deepseek-vl2",
                     help="deepseek-ai/deepseek-vl2 (27B), deepseek-vl2-small (16B) or deepseek-vl2-tiny (3B)")
_parser.add_argument("--no-p2p", action="store_true",
                     help="Always copy tensors between GPUs through CPU memory (auto-enabled if the startup check fails)")
_args = _parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = _args.device
# Stop transformers from importing TensorFlow (installed but broken in this env).
os.environ["USE_TF"] = "0"

import math
import traceback
from collections.abc import Mapping
import torch
import re
import json

import accelerate.hooks
from accelerate.utils import honor_type
from transformers import AutoModelForCausalLM
from deepseek_vl2.models import DeepseekVLV2Processor
from deepseek_vl2.utils.io import load_pil_images


device = torch.device("cuda:0")

CONFIDENCE_THRESHOLD = 0.25


model_path = _args.model

# Decoder layer count per checkpoint, used to split the language model over GPUs.
NUM_LAYERS = {
    "deepseek-ai/deepseek-vl2": 30,
    "deepseek-ai/deepseek-vl2-small": 27,
    "deepseek-ai/deepseek-vl2-tiny": 12,
}


def split_model(num_layers, num_gpus):
    # GPU 0 also holds the vision tower, projector, embeddings, norm and lm_head,
    # so it takes about half as many decoder layers as each other GPU.
    per_gpu = math.ceil(num_layers / (num_gpus - 0.5))
    counts = [math.ceil(per_gpu * 0.5)] + [per_gpu] * (num_gpus - 1)

    device_map = {}
    layer = 0
    for gpu, count in enumerate(counts):
        for _ in range(count):
            if layer < num_layers:
                device_map[f"language.model.layers.{layer}"] = gpu
                layer += 1

    for name in ["vision", "projector", "image_newline", "view_seperator",
                 "language.model.embed_tokens", "language.model.norm", "language.lm_head"]:
        device_map[name] = 0
    # Same as DeepSeek-VL2's own multi-GPU loader: last layer sits next to norm/lm_head.
    device_map[f"language.model.layers.{num_layers - 1}"] = 0
    return device_map


processor = DeepseekVLV2Processor.from_pretrained(model_path)

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

print(f"Splitting {model_path} across {num_gpus} GPU(s): CUDA_VISIBLE_DEVICES={_args.device}")

# device_map shards the weights across GPUs (accelerate moves activations
# between them), so no .to(device) here. Inputs still go to cuda:0.
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map=split_model(NUM_LAYERS[model_path], num_gpus),
).eval()


image_folder = _args.images
json_output_path = _args.output

os.makedirs(os.path.dirname(json_output_path) or ".", exist_ok=True)


original_prompt = (
    "You are an animal body-part detector. Carefully examine this image and count ONLY the parts that are clearly visible.\n\n"
    "Count these 7 parts:\n"
    "(a) Head - the animal head. For a clearly visible animal, this is usually 1.\n"
    "(b) Eyes - visible animal eyes. If both eyes are clearly visible, count 2. If only one eye is visible, count 1. If none are visible, count 0.\n"
    "(c) Legs - visible animal legs. Count each clearly visible leg. Some animals may have 2(if it's a bird) or 4 visible legs depending on pose and occlusion.\n"
    "(d) Tail - visible animal tail. Usually 1 if visible.\n"
    "(e) Torso - main body of the animal. Usually 1 if visible.\n"
    "(f) Beak - ONLY for birds. Count 1 if clearly visible, otherwise 0.\n"
    "(g) Wings - ONLY for birds. If both wings are visible, count 2. If only one wing is visible, count 1. If none are visible, count 0.\n\n"
    "IMPORTANT RULES:\n"
    "(h) Foreign parts: Count any features clearly visible that do NOT belong on this animal "
    "(e.g. human hands, human face, mechanical parts, wings on a mammal, trunk on a bird). "
    "Count 0 if none are present.\n"    
    "- Only count parts that are clearly visible in the image.\n"
    "- Do NOT assume hidden or occluded parts exist.\n"
    "- If you cannot see a component clearly, count it as 0.\n"
    "- Base all counts only on visible evidence.\n"
    "- Use the most likely visible count, not a guess.\n\n"
    "Respond EXACTLY in this format:\n"
    "(a) Head: [number]\n"
    "(b) Eyes: [number]\n"
    "(c) Legs: [number]\n"
    "(d) Tail: [number]\n"
    "(e) Torso: [number]\n"
    "(f) Beak: [number]\n"
    "(g) Wings: [number]\n"
    "(h) Foreign parts: [number]\n"
    "Overall confidence: [0.0-1.0]"
)


def extract_counts(text):

    mapping = {
        "head": r"head:\s*(\d+)",
        "eye": r"eyes?:\s*(\d+)",
        "leg": r"legs?:\s*(\d+)",
        "tail": r"tail:\s*(\d+)",
        "torso": r"torso:\s*(\d+)",
        "beak": r"beak:\s*(\d+)",
        "wing": r"wings?:\s*(\d+)",
        "foreign_part": r"foreign\s+parts?:\s*(\d+)"
    }

    res = {}

    for k, p in mapping.items():
        matches = re.findall(p, text.lower())
        res[k] = int(matches[-1]) if matches else 0

    return res


def extract_conf(text):

    matches = re.findall(r"confidence[^\d]*(\d*\.?\d+)", text.lower())

    if not matches:
        return 0.0

    try:
        val = float(matches[-1])
        return val if 0 <= val <= 1.0 else 0.0
    except ValueError:
        return 0.0


image_files = sorted([
    f for f in os.listdir(image_folder)
    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
])

results_list = []
failures = 0


for image_file in image_files:

    conversation = [
        {
            "role": "user",
            "content": f"<image>\n{original_prompt}",
            "images": [os.path.join(image_folder, image_file)]
        },
        {
            "role": "assistant",
            "content": ""
        }
    ]

    try:

        pil_images = load_pil_images(conversation)

        inputs = processor(
            conversations=conversation,
            images=pil_images,
            force_batchify=True
        ).to(device)

        with torch.no_grad():
            emb = model.prepare_inputs_embeds(**inputs)
            # Call the language model directly, as in DeepSeek-VL2's README. Passing only
            # inputs_embeds means generate() needs the special token ids, else it raises
            # "`bos_token_id` has to be defined when no `input_ids` are provided".
            outputs = model.language.generate(
                inputs_embeds=emb,
                attention_mask=inputs.attention_mask,
                pad_token_id=processor.tokenizer.eos_token_id,
                bos_token_id=processor.tokenizer.bos_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
                max_new_tokens=512,
                do_sample=False,
                use_cache=True
            )

        answer = processor.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        components = extract_counts(answer)
        confidence = extract_conf(answer)

        results_list.append({
            "image_id": image_file,
            "components": components,
            "leg_count": components.get("leg", 0),
            "average_confidence": round(confidence, 2),
            "total_detections": sum(components.values()),
            "detections": []
        })

        print(f"Processed: {image_file} | Conf: {confidence}")

    except Exception as e:
        print(f"Error {image_file}: {e}")
        # Full traceback once, so the failing library is visible.
        if failures == 0:
            traceback.print_exc()
        failures += 1
        # A CUDA assert leaves the GPUs unusable for the rest of this process.
        if "CUDA error" in str(e):
            raise SystemExit("Stopping: CUDA error, every later image would fail the same way.")


# Don't overwrite an existing results file when every image failed.
if not results_list:
    raise SystemExit(f"All {len(image_files)} images failed; not writing {json_output_path}")


final_output = {
    "model": "Deepseek-VL2",
    "confidence_threshold": CONFIDENCE_THRESHOLD,
    "total_images": len(image_files),
    "total_detections": sum(r["total_detections"] for r in results_list),
    "results": results_list
}


with open(json_output_path, "w") as f:
    json.dump(final_output, f, indent=2)