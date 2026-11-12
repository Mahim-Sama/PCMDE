# PCMDE: Physical-Context-based Multimodal Data Evaluation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](#)
[![CUDA 12.2+](https://img.shields.io/badge/CUDA-12.2+-green.svg)](#)

PCMDE is a domain-agnostic framework for evaluating whether synthetic images are
*physically plausible* — not merely whether they match their caption. It detects
object components with an ensemble of a CNN/transformer detector and two vision
language models, then checks those detections against physical rules written in
JSON, and finally asks an LLM for an independent verdict.

An aircraft engine floating above the fuselage, a car with wheels above the
roofline, or a dog with no torso each trigger a violation regardless of how well
the image matches its caption.

Paper: available on arXiv.

---

## Pipeline at a glance

```text
                Stage 1: multi-source detection
  images ──┬──▶ detector (YOLOv12 / Grounding DINO+SAM) ──┐
           ├──▶ DeepSeek-VL2  ────────────────────────────┤
           └──▶ Pixtral-12B   ────────────────────────────┤
                                                          ▼
                                        pipeline/ensemble.py  (confidence-weighted voting)
                                                          │
                                                          ▼  ensemble JSON
                          Stage 2: rule engine ◀───── domains/<d>/physical_rules_*.json
                                     │
                          Stage 3: LLM reasoner (deepseek-r1:70b via Ollama)
                                     │
                                     ▼
                     S_final = 0.6·S_rules + 0.4·S_LLM  +  PASS/FAIL + diagnostics
```

## Repository layout

```text
core/                 Rule engine, LLM validator, scoring. Domain-agnostic.
cli/evaluate.py       Entry point for Stages 2+3.
configs/              One YAML per domain: aircraft, animal, car.
domains/<d>/          Physical rules JSON per domain (all domain knowledge).
pipeline/
  detection/          YOLOv12 training + inference notebook.
  vlm/                DeepSeek-VL2 and Pixtral-12B inference, per domain.
  ensemble.py         Fuses detector + 2 VLM outputs into one ensemble JSON.
                      --mode also builds the detection-ablation inputs.
metrics/              Baseline cross-check: CLIPScore, SigLIP-2, VQAScore, FLEUR.
analysis/
  check_rule_vocabulary.py  Do the rules and the detector speak the same vocabulary?
  run_ablation.py           Produces the five ablation variants for one domain.
  ablation_analysis.py      Ablation variants -> Table IV / XIII rows (text + LaTeX).
  ablation_analysis_{aircraft,car}.py   Legacy, reads the 2026-03 output schema.
data/<d>/             images/ (empty, you supply) · captions.csv · detections/
                      vlm/ · ensemble/ · generated_images_log.csv
results/<d>/          Run output, created on the first run: scores/ ·
                      ablation/ · crosscheck/ · detections_annotated/
scripts/run_domain.sh One-domain driver: fusion -> vocab -> evaluate -> ablation.
```

---

## Installation

Requirements are split by stage so you don't need a GPU to reproduce scores from
existing ensembles.

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements/core.txt        # Stages 2+3 + analysis (CPU only)
pip install -r requirements/detection.txt   # Stage 1a (GPU)
pip install -r requirements/vlm.txt         # Stage 1b (large-VRAM GPU)
pip install -r requirements/metrics.txt     # baseline cross-check (GPU)
```

Use **separate environments** for `vlm.txt` and `metrics.txt`. DeepSeek-VL2,
Pixtral, VQAScore and FLEUR pin conflicting `transformers` versions; the
requirements files note which combinations clash.

Full-pipeline hardware: Python 3.10+, CUDA 12.2+, ~64 GB RAM, ~80 GB disk for
weights, 80 GB VRAM recommended to host the VLMs and the LLM reasoner together.

### LLM reasoner (Stage 3)

```bash
ollama pull deepseek-r1:70b
ollama serve
```

Endpoint is `llm.api_base` in each config (default `http://localhost:11434`).
Pass `--no-llm` to skip Stage 3 entirely.

### Getting the test images

You supply the test images (~330 MB). Each domain expects 100 images.

`data/<domain>/captions.csv` is the filename list. The column holding the
filename differs per domain -- it is `csv_mapping.image_col` in the domain's
config: `Image_path` for aircraft and animal, `image_id` for car. The same
names appear in the detection, VLM and ensemble JSONs.

```bash
# aircraft / animal
python -c "import csv;[print(r['Image_path']) for r in csv.DictReader(open('data/animal/captions.csv',encoding='utf-8-sig'))]"
# car
python -c "import csv;[print(r['image_id']) for r in csv.DictReader(open('data/car/captions.csv',encoding='utf-8-sig'))]"
```

One known discrepancy: `data/aircraft/captions.csv` holds 101 rows, and one of
them (`nsid_124.png`) has no image on disk and no entry in the ensemble. The
aircraft set is 100 images; treat that row as spurious.

Place the files at `data/<domain>/images/`. Everything except image-reading
steps works without them.

---

## Quick start

Score a domain end-to-end from an existing ensemble — no GPU, no images:

```bash
# Rules only (no Ollama needed)
python cli/evaluate.py --config configs/animal.yaml --no-llm

# Full pipeline (Ollama must be running)
python cli/evaluate.py --config configs/animal.yaml
```

**Always run from the project root** — config paths resolve against the working
directory.

Overrides:

```bash
python cli/evaluate.py --config configs/car.yaml \
    --ensemble data/car/ensemble/ensemble_car.json \
    --captions data/car/captions.csv \
    --output   results/car/scores/car_pcmde_scores.json
```

---

## Running the full experiment

Every command is per domain; `<domain>` is `aircraft`, `animal` or `car`. Run
from the project root. Stage 1 needs GPUs, Stages 2+3 need Ollama.

### Stage 1a — Detection

```bash
jupyter notebook pipeline/detection/yolov12_train_infer.ipynb   # car, aircraft
```

```bash
python pipeline/detection/groundingdino_sam_animal.py --force        --annotate results/animal/detections_annotated               # animal
```

Output goes to `data/<domain>/detections/`.

### Stage 1b — VLM inference

One script per model per domain (6 total). Paths and GPU are options, defaulting
to the repo layout:

```bash
python pipeline/vlm/deepseek_<domain>.py --images data/<domain>/images \
       --output data/<domain>/vlm/deepseek_<domain>.json --device 0
python pipeline/vlm/pixtral_<domain>.py  --images data/<domain>/images \
       --output data/<domain>/vlm/pixtral_<domain>.json  --device 1
```

### Stage 1c — Ensemble fusion

```bash
python pipeline/ensemble.py --config configs/<domain>.yaml
```

Reads `paths.detections_json` + `paths.vlm_jsons` and writes
`paths.ensemble_json`. The explicit form still works:

```bash
python pipeline/ensemble.py \
    data/animal/detections/groundingdino_sam_animal.json \
    data/animal/vlm/deepseek_animal.json \
    data/animal/vlm/pixtral_animal.json \
    --rules domains/animal/physical_rules_animal_v2.json \
    -o data/animal/ensemble/ensemble_animal.json
```

The detector is auto-identified as the file carrying bounding boxes
(`--detector-file` forces it); the remaining files become VLM-1 and VLM-2 **in
the order you pass them**, which is what the 0.32 / 0.18 weights follow. Part
names are normalised and mapped through the `component_aliases` block of the
rules JSON, so `Wheels`, `wheels` and `bonnet (front hood)` all land on one
component. `--mode detector_only|vlm_only` builds the ablation inputs.

### Stages 2+3 — Evaluation

```bash
python cli/evaluate.py --config configs/aircraft.yaml
python cli/evaluate.py --config configs/animal.yaml
python cli/evaluate.py --config configs/car.yaml
```

Output per image: overall score, PASS/FAIL verdict, rule and LLM sub-scores,
per-category breakdown, the specific rules violated, rules that could not be
evaluated, and the LLM's free-text reasoning.

If the LLM stage fails for an image (timeout, empty reply, a reply truncated
inside `<think>`), the run says so: the image is flagged `llm_failed`, counted in
`statistics.llm_failures`, and its LLM score mirrors the rule score instead of an
invented value. **A run with failures is not reportable** — fix the LLM setup and
re-run those domains.

Runs write to `results/<domain>/scores/`.

### Ablation (paper Sec. VI-F)

```bash
python analysis/run_ablation.py --config configs/<domain>.yaml
python analysis/ablation_analysis.py --config configs/aircraft.yaml configs/animal.yaml configs/car.yaml
```

`run_ablation.py` builds the detector-only and VLM-only ensembles, evaluates
those two plus the full pipeline, and derives rules-only and LLM-only by
re-weighting the full run (1.0/0.0 and 0.0/1.0). It writes five variant JSONs, a
manifest and a per-image CSV into `paths.ablation_dir`. Budget three LLM passes
over the dataset per domain; `--variants` and `--force` control re-runs, and
`--no-llm` is a plumbing test only.

`ablation_analysis.py` prints Table IV (PCMDE rows) and Table XIII (a) and (b),
as text and as LaTeX rows, and warns when a run had LLM failures, ran without the
LLM, or covers different image counts across variants.

### Baseline cross-check

```bash
python metrics/crosscheck.py --config configs/<domain>.yaml --metric all   # CLIPScore + SigLIP-2
python metrics/fleur.py      --config configs/<domain>.yaml                # FLEUR (needs ./LLaVA checkout)
python metrics/crosscheck_summary.py --config configs/aircraft.yaml configs/animal.yaml configs/car.yaml
```

`crosscheck.py` and `fleur.py` read paths and caption columns from the domain
config and write to `results/<domain>/crosscheck/`, keeping the filenames and
schema of the May 2026 animal run. `crosscheck_summary.py` turns those files,
plus the PCMDE scores, into the Table IV dispersion rows (text + LaTeX) and
names any metric not yet run.

VQAScore runs in its own project/environment (`t2v-metrics`); save its per-image
JSON as `results/<domain>/crosscheck/vqascore_<domain>.json` (or pass
`--vqascore <path>`) and the summary picks it up.
`metrics/other_metrics_crosscheck.ipynb` is the original notebook those scripts
came from and still has hardcoded HPC car paths.

### Sanity checks

```bash
python analysis/check_rule_vocabulary.py --config configs/<domain>.yaml   # before any run
```

`check_rule_vocabulary.py` verifies that every component your rules reference is
actually emitted by the detector, and flags case/plural collisions. A
`critical`-severity rule pointing at a component the detector never produces caps
*every* image at `critical_violation_cap` and silently makes the whole run
meaningless. Run it after authoring a new domain or changing a detector's class
list. All three domains currently report `RESULT: OK`.

---

## Dataset and results

Each domain has **100 test images** generated with HiDream-I1. Seeds, prompts
and defect flags for the generated images are logged in
`data/<domain>/generated_images_log.csv`.

Verified from `results/`:

- **Stages 1a -> 1c**: the ensembles carry 100 images per domain.
- **Stages 2+3**: `statistics.llm_failures` is **0** for all three domains.
  Mean overall: aircraft 72.68, animal 85.19, car 85.09.
- **Ablation** for all three domains (five variants, manifest and per-image CSV
  in `results/<domain>/ablation/`).
- **Baseline cross-check** for all three domains -- CLIPScore, SigLIP-2, FLEUR
  and VQAScore, plus `results/crosscheck_summary.csv`.

Rules worth reviewing, as they fire widely:

- animal head-position tolerance (`extremum.head.x_either.18`): fails on
  **36/100** images;
- car grille count (`count_range.grille.11`, paper says exactly 1): fails on
  **27/100** images;
- `size_ratio.eye_width_over_head_width.26` (39/100, animal) and
  `size_ratio.head_width_over_torso_width.default.22` (27/100, animal).

---

## Adding a new domain

One JSON and one YAML. Zero Python changes.

1. Author `domains/<name>/physical_rules_<name>.json` — copy an existing
   domain's file and edit it; every rule carries its own predicate and severity.
2. Create `configs/<name>.yaml` by copying an existing one and repointing paths.
3. Put data under `data/<name>/{images,detections,vlm,ensemble}` and `captions.csv`.
4. `python analysis/check_rule_vocabulary.py --config configs/<name>.yaml` — confirm
   it reports `RESULT: OK` before going further.
5. `python cli/evaluate.py --config configs/<name>.yaml`

The rules JSON is self-describing: every rule carries its own predicate, parameters
and severity. No vocabulary or knowledge file is needed.

---

## Scoring summary

| Category | Meaning |
|---|---|
| Presence | Are required components detected in valid quantities? |
| Spatial | Are components positioned correctly relative to each other? |
| Relational | Do component dependencies and proportions hold? |
| Subtype | Are components consistent with the caption's declared subtype? |

`S_final = 0.6·S_rules + 0.4·S_LLM`. **PASS** requires the overall score ≥ 60 *and*
both components ≥ 40. A `critical` severity violation caps the rule score.

Severity levels: `critical` (obviously unusable, high certainty) → `major`
(structurally wrong, partially usable) → `minor` (imprecision, still plausible).
