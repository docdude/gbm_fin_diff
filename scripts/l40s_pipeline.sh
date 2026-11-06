#!/usr/bin/env bash
# Full run on a single GPU (auto-selects an L40S if present).
#
#   1. Train this repo (fixed): GBM + exponential, paper architecture, 1000 epochs
#   2. Optional ablation: same with the paper's σ_min = 0.01 (RUN_ABLATION=1)
#   3. Train + sample vnhtg01/gbm-diffusion-finance-ts on the same tickers
#   4. Score everything with the shared harness → save/comparison/REPORT.md
#
# Usage:
#   bash scripts/l40s_pipeline.sh                  # main comparison
#   RUN_ABLATION=1 bash scripts/l40s_pipeline.sh   # + σ_min=0.01 ablation
#   GPU=1 bash scripts/l40s_pipeline.sh            # force a GPU index
#   FOREGROUND=1 bash scripts/l40s_pipeline.sh     # stay attached to the terminal
#
# By default the script detaches itself (setsid + nohup) and only writes to
# log files, so closing the SSH/VS Code terminal cannot stall or kill it.
# Follow progress with:  tail -f save/logs/pipeline.out
#
# Every stage is resumable: re-running skips finished stages and resumes
# training from the newest checkpoint_epoch*.pth (LR schedule included).
# Prerequisite: data/financial/sp500_prices_raw.pkl (copied from the dev box,
# or downloaded automatically on first run if yfinance has network access).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p save/logs

if [[ "${FOREGROUND:-0}" != "1" && -z "${_PIPELINE_DETACHED:-}" ]]; then
  if pgrep -f "gbm_financial.train_l4|compare_vnhtg01.py" >/dev/null; then
    echo "A pipeline stage is already running:"; pgrep -af "gbm_financial.train_l4|compare_vnhtg01.py"
    exit 1
  fi
  _PIPELINE_DETACHED=1 setsid nohup bash "$0" "$@" >> save/logs/pipeline.out 2>&1 < /dev/null &
  echo "Pipeline started in the background (PID $!). Safe to close this terminal."
  echo "Follow: tail -f save/logs/pipeline.out"
  exit 0
fi

PY=${PY:-python}
EPOCHS=${EPOCHS:-1000}
THEIR_EPOCHS=${THEIR_EPOCHS:-1000}
RUN_ABLATION=${RUN_ABLATION:-0}

if [[ -z "${GPU:-}" ]]; then
  GPU=$(nvidia-smi --query-gpu=index,name --format=csv,noheader | grep -i "L40S" | head -1 | cut -d, -f1 || true)
  GPU=${GPU:-0}
fi
export CUDA_VISIBLE_DEVICES=$GPU
echo "Using GPU $GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader -i "$GPU")"

echo "=== Pipeline start $(date) ==="

run_ours () {  # $1 = save_dir, rest = extra args
  local dir=$1; shift
  if [[ -f "$dir/gbm_exponential/results.json" ]]; then
    echo "Skip: $dir already finished"
    return
  fi
  local resume=()
  local last
  last=$(ls "$dir"/gbm_exponential/checkpoint_epoch*.pth 2>/dev/null | sort -V | tail -1 || true)
  [[ -n "$last" ]] && resume=(--resume "$last") && echo "Resuming from $last"
  $PY -m gbm_financial.train_l4 --save_dir "$dir" --sde_type gbm --schedule exponential \
      --epochs "$EPOCHS" "${resume[@]}" "$@" >> "save/logs/$(basename "$dir").log" 2>&1
}

echo "=== [1] This repo (fixed) ==="
run_ours save/fixed

EXTRA=()
if [[ "$RUN_ABLATION" == "1" ]]; then
  echo "=== [2] Ablation: paper σ_min = 0.01 ==="
  run_ours save/ablation_sigmamin001 --sigma_min 0.01
  EXTRA=(--ours-extra "ours σ_min=0.01 (paper)=save/ablation_sigmamin001/gbm_exponential")
fi

echo "=== [3-4] vnhtg01 + report ==="
$PY scripts/compare_vnhtg01.py all --ours save/fixed/gbm_exponential \
    --their-epochs "$THEIR_EPOCHS" "${EXTRA[@]}" >> save/logs/compare_vnhtg01.log 2>&1

echo "=== Done $(date) → save/comparison/REPORT.md ==="
