#!/bin/bash
# 只跑玩具 A 冒烟（第 3、4 步），用于改了 toyA_audit/analyze 之后快速复查。TRUTH 可临时替代真值，只验证流程。
set -euo pipefail
D=/workspace/go-toy; O=/workspace/go-oldnet-conv
PY=/workspace/acupuncture/.venv-bench/bin/python
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=${THREADS:-2} MKL_NUM_THREADS=${THREADS:-2}
export KATAGO_PY=$O/KataGo/python OLDNET_SCRIPTS=$O/scripts TOY_PYLIB=$D/pylib
T=$D/toy; N=$D/nets
H=$N/kata1-b18c384nbt-s9311982848-d4134175181/model.ckpt
NB=$N/kata1-b18c384nbt-s8859591680-d4016192437/model.ckpt
E=$N/kata1-b18c384nbt-s4975305984-d3174897359/model.ckpt
TR=${TRUTH:-$N/kata1-b28c512nbt-s13255194368-d5935380940/model.ckpt}
B40=$O/out/b40_fixup_tf/model.ckpt; B10=$O/out/b10_fixup/model.ckpt
TAG=${TAG:-A}
cd $T
date
nice -n 19 $PY -I toyA_audit.py --host $H --neighbor $NB --auditors $H,$E,$B40,$B10 --names self,b18e,b40,b10 --truth $TR \
  --trigger $D/out/trigger_smoke.json --sgf "$D/data/sgfs/*/*/*.sgf" --n 16 --seed 2 --h1 0.10 --h-min 0.02 --ks 0.9,0.99 \
  --h-min-hi 0.05 --games 1 --top-k-f1 20 --pool-mult 60 --device cpu --out $D/out/toyA_smoke_$TAG.jsonl
date
nice -n 19 $PY -I analyze.py $D/out/toyA_smoke_$TAG.jsonl --M 6 --E 2,4 --E-gate 4 --B 500 --json $D/out/toyA_smoke_${TAG}_summary.json
echo smokeA_done
