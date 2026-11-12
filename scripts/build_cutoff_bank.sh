#!/usr/bin/env bash
# Build a synthetic scenario bank whose generators never see data after CUTOFF.
#
# Use when the DRL side selects on a validation tail inside the generator's
# default training range (optuna_vec_mace_v2 carves the last 15% of TRAIN:
# 2022-04-14 → 2024-05-31), so synthetic training data cannot leak it.
#
#   CUTOFF=2022-04-13 GPUS="0 1" setsid nohup scripts/build_cutoff_bank.sh > save/logs/cutoff_bank.log 2>&1 &
#
# Steps (≈7 h on two GPUs, ≈14 h on one): train market/ticker EDM + ε models,
# 2×10 scenarios (all corrections on), panel_eval, FinRL-Meta env export.
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

COMMON=(--n_scenarios 10 --end_date "$CUTOFF"
        --ticker_dir "$OUT/ticker_edm" --ticker_fine "$OUT/ticker_eps/best_model.pth"
        --market_dir "$OUT/market_edm" --market_fine "$OUT/market_eps/best_model.pth")
for k in 0 1; do
  g=${GPUS[$(( k % ${#GPUS[@]} ))]}
  CUDA_VISIBLE_DEVICES=$g $PY -m gbm_financial.panel_assembly --seed $k --out "$OUT/scenarios_s$k" \
    "${COMMON[@]}" > "save/logs/cut_${CUTOFF}_scen_s$k.log" 2>&1 &
  [ ${#GPUS[@]} -ge 2 ] || wait
done
wait
mkdir -p "$OUT/scenarios"; i=0
for k in 0 1; do for j in $(seq -f "%03g" 0 9); do n=$(printf "%03d" $i)
  for suf in .csv.gz _market.csv _raw.npz; do cp "$OUT/scenarios_s$k/scenario_$j$suf" "$OUT/scenarios/scenario_$n$suf"; done
  i=$((i + 1)); done; done
$PY -m gbm_financial.panel_eval --scenarios "$OUT/scenarios" --end_date "$CUTOFF" > "save/logs/cut_${CUTOFF}_panel_eval.log" 2>&1
"$FINRL_META/.venv/bin/python" scripts/export_finrl_meta_env.py --scenarios "$OUT/scenarios" \
  --finrl_meta "$FINRL_META" > "save/logs/cut_${CUTOFF}_export.log" 2>&1
echo "done $(date): $OUT/scenarios/*.env.pkl"
