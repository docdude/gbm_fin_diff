#!/usr/bin/env bash
# Build a synthetic scenario bank whose generators never see data after CUTOFF.
#
# Use when the DRL side selects on a validation tail inside the generator's
# default training range (optuna_vec_mace_v2 carves the last 15% of TRAIN:
# 2022-04-14 → 2024-05-31), so synthetic training data cannot leak it.
#
#   CUTOFF=2022-04-13 GPUS="0 1" setsid nohup scripts/build_cutoff_bank.sh > save/logs/cutoff_bank.log 2>&1 &
#
# Steps (≈7 h on two GPUs, ≈14 h on one): train market/ticker EDM + ε models, then
# scripts/build_banks.sh (regime-stratified train + eval banks, eval, env export;
# ≈6 h more on two GPUs; sizes/mixes via TRAIN_N, EVAL_N, TRAIN_MIX, EVAL_MIX).
set -euo pipefail
cd "$(dirname "$0")/.."
CUTOFF=${CUTOFF:-2022-04-13}
GPUS=(${GPUS:-0})
OUT=${OUT:-save/panel/cut_${CUTOFF}}
FINRL_META=${FINRL_META:-$HOME/FinRL-Meta}
PY=.venv/bin/python
mkdir -p "$OUT" save/logs
export CUDA_DEVICE_ORDER=PCI_BUS_ID
echo "cutoff bank: CUTOFF=$CUTOFF GPUS=${GPUS[*]} OUT=$OUT  $(date)"

train() {  # gpu model precond name
  local extra=(); [ "$3" = edm ] && extra=(--precond edm)
  CUDA_VISIBLE_DEVICES=$1 $PY -m gbm_financial.train_panel --model "$2" "${extra[@]}" \
    --end_date "$CUTOFF" --save_dir "$OUT/$4" > "save/logs/cut_${CUTOFF}_$4.log" 2>&1
}
JOBS=("ticker edm ticker_edm" "ticker eps ticker_eps" "market edm market_edm" "market eps market_eps")
if [ ${#GPUS[@]} -ge 2 ]; then
  (train "${GPUS[0]}" ${JOBS[0]}; train "${GPUS[0]}" ${JOBS[2]}) &
  (train "${GPUS[1]}" ${JOBS[1]}; train "${GPUS[1]}" ${JOBS[3]}) &
  wait
else
  for j in "${JOBS[@]}"; do train "${GPUS[0]}" $j; done
fi
echo "models trained $(date)"

MODELS="$OUT" CUTOFF="$CUTOFF" GPUS="${GPUS[*]}" FINRL_META="$FINRL_META" scripts/build_banks.sh
echo "done $(date): $OUT/bank_train/scenarios, $OUT/bank_eval/scenarios"
