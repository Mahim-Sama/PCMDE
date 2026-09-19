#!/usr/bin/env bash
# Stage 1c -> vocabulary check -> Stages 2+3 -> ablation, for one domain.
#
#   bash scripts/run_domain.sh car                # all steps
#   bash scripts/run_domain.sh car evaluate       # resume at: ensemble | vocab | evaluate | ablation
#
# Logs go to logs/<domain>_<n>_<step>.log. Run inside the pcmdenv environment.
set -u
d=${1:?usage: bash scripts/run_domain.sh <car|aircraft|animal> [ensemble|vocab|evaluate|ablation]}
start=${2:-ensemble}
case "$start" in ensemble|vocab|evaluate|ablation) ;; *) echo "unknown step: $start"; exit 2 ;; esac

cd "$(dirname "$0")/.." || exit 1
mkdir -p logs
scores="results/$d/scores/${d}_pcmde_scores.json"
running=0
step () { [ "$1" = "$start" ] && running=1; [ $running -eq 1 ]; }

# Every run reaches the LLM steps; fail now rather than after fusion.
curl -sf localhost:11434/api/tags >/dev/null || { echo "Ollama is not reachable on localhost:11434"; exit 1; }

if step ensemble; then
  echo "===== $d : fusion ====="
  python pipeline/ensemble.py --config configs/$d.yaml 2>&1 | tee logs/${d}_1_ensemble.log
  [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "FAILED: ensemble ($d)"; exit 1; }
fi

if step vocab; then
  echo "===== $d : vocabulary check ====="
  python analysis/check_rule_vocabulary.py --config configs/$d.yaml 2>&1 | tee logs/${d}_2_vocab.log
  [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "FAILED: vocabulary check ($d)"; exit 1; }
fi

if step evaluate; then
  echo "===== $d : full pipeline with LLM ($(date)) ====="
  python cli/evaluate.py --config configs/$d.yaml 2>&1 | tee logs/${d}_3_evaluate.log
  [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "FAILED: evaluate ($d)"; exit 1; }
  python - "$scores" <<'PY' || { echo "LLM failures in $d. Fix Ollama, then: bash scripts/run_domain.sh $d evaluate"; exit 1; }
import json, sys
s = json.load(open(sys.argv[1]))["statistics"]
print("statistics:", s)
sys.exit(1 if s.get("llm_failures") else 0)
PY
fi

if step ablation; then
  echo "===== $d : ablation ($(date)) ====="
  [ -f "$scores" ] || { echo "Missing $scores - run the evaluate step first"; exit 1; }
  mkdir -p results/$d/ablation
  # Ablation files older than the full run are leftovers (e.g. a 70-image dry run).
  # Newer ones come from an interrupted ablation of this run and are reused.
  find results/$d/ablation -maxdepth 1 -type f \
       \( -name "ensemble_${d}_*.json" -o -name "${d}_ablation_*.json" -o -name "${d}_per_image_scores.csv" \) \
       ! -newer "$scores" -print -delete
  cp "$scores" results/$d/ablation/${d}_ablation_full_pipeline.json
  python analysis/run_ablation.py --config configs/$d.yaml 2>&1 | tee logs/${d}_4_ablation.log
  [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "FAILED: ablation ($d)"; exit 1; }
  if grep -q "had no valid LLM reply" logs/${d}_4_ablation.log; then
    echo "LLM failures in the $d ablation. Re-run the failing variant, then resume:"
    echo "  python analysis/run_ablation.py --config configs/$d.yaml --variants vlm_only --force   # or detector_only"
    echo "  bash scripts/run_domain.sh $d ablation"
    exit 1
  fi
fi

echo "===== $d : done ($(date)) ====="
