#!/usr/bin/env bash
# Build the DRL scenario banks from trained panel models.
#
#   train bank: long-run realistic regime mix (1995→cutoff NAS-100 frequencies)
#   eval  bank: balanced regimes, separate seeds — per-regime robustness testing only
#
#   MODELS=save/panel/cut_2022-04-13 CUTOFF=2022-04-13 GPUS="0 1" scripts/build_banks.sh
#
# Bear/crash paths may come from the market model's full 1995+ history
# (--stress_full_history); bull/sideways paths keep the 2010+ era selection.
set -euo pipefail
cd "$(dirname "$0")/.."
MODELS=${MODELS:?set MODELS=<dir with ticker_edm ticker_eps market_edm market_eps>}
CUTOFF=${CUTOFF:-2024-05-31}
GPUS=(${GPUS:-0})
TRAIN_N=${TRAIN_N:-60}
EVAL_N=${EVAL_N:-20}
TRAIN_MIX=${TRAIN_MIX:-bull=0.45,sideways=0.21,bear=0.10,crash=0.24}
EVAL_MIX=${EVAL_MIX:-balanced}
FINRL_META=${FINRL_META:-$HOME/FinRL-Meta}
PY=.venv/bin/python
export CUDA_DEVICE_ORDER=PCI_BUS_ID
NG=${#GPUS[@]}
echo "banks: MODELS=$MODELS CUTOFF=$CUTOFF GPUS=${GPUS[*]} train $TRAIN_N ($TRAIN_MIX) eval $EVAL_N ($EVAL_MIX)  $(date)"

split_mix() {  # mix n parts → one exact count spec per line
  $PY - "$1" "$2" "$3" <<'EOF'
import sys
from gbm_financial.panel_assembly import parse_regime_mix
q = parse_regime_mix(sys.argv[1], int(sys.argv[2])); parts = int(sys.argv[3])
cnt = [dict.fromkeys(q, 0) for _ in range(parts)]
i = 0
for k in q:
    for _ in range(q[k]):
        cnt[i % parts][k] += 1; i += 1
for c in cnt:
    print(sum(c.values()), ",".join(f"{k}={v}" for k, v in c.items()))
EOF
}

bank() {  # name n mix seed_base
  local name=$1 n=$2 mix=$3 seed=$4 out="$MODELS/bank_$1" k=0
  mkdir -p "$out"
  while read -r cnt spec; do
    local g=${GPUS[$(( k % NG ))]}
    CUDA_VISIBLE_DEVICES=$g $PY -m gbm_financial.panel_assembly --seed $(( seed + k )) \
      --out "$out/part$k" --n_scenarios "$cnt" --regime_mix "$spec" --stress_full_history \
      --end_date "$CUTOFF" --ticker_dir "$MODELS/ticker_edm" --ticker_fine "$MODELS/ticker_eps/best_model.pth" \
      --market_dir "$MODELS/market_edm" --market_fine "$MODELS/market_eps/best_model.pth" \
      > "save/logs/bank_${name}_part$k.log" 2>&1 &
    k=$((k + 1))
  done < <(split_mix "$mix" "$n" "$NG")
  wait
  mkdir -p "$out/scenarios"; local i=0
  for p in "$out"/part*; do
    for f in "$p"/scenario_*[0-9].csv.gz; do
      local b; b=$(basename "$f" .csv.gz); local m; m=$(printf "scenario_%03d" $i)
      cp "$f" "$out/scenarios/$m.csv.gz"
      cp "$p/${b}_market.csv" "$out/scenarios/${m}_market.csv"
      cp "$p/${b}_raw.npz" "$out/scenarios/${m}_raw.npz"
      i=$((i + 1))
    done
  done
  $PY -m gbm_financial.panel_eval --scenarios "$out/scenarios" --end_date "$CUTOFF" > "save/logs/bank_${name}_eval.log" 2>&1
  "$FINRL_META/.venv/bin/python" scripts/export_finrl_meta_env.py --scenarios "$out/scenarios" \
    --finrl_meta "$FINRL_META" > "save/logs/bank_${name}_export.log" 2>&1
  $PY scripts/characterize_scenarios.py --scenarios "$out/scenarios" --end_date "$CUTOFF" --plot \
    > "save/logs/bank_${name}_regimes.log" 2>&1
  echo "bank $name done $(date): $out/scenarios ($i scenarios)"
}

bank train "$TRAIN_N" "$TRAIN_MIX" 1000
bank eval "$EVAL_N" "$EVAL_MIX" 2000
echo "all banks done $(date)"
