#!/bin/bash
# T0 迷你合同的运行脚本。用法：
#   bash toy/run_t0.sh select            # 重选触发器 → out/t0/trigger_t0.json
#   bash toy/run_t0.sh smoke             # 6 个局面计时冒烟（不作证据）→ out/t0/smoke.jsonl
#   bash toy/run_t0.sh shard <i> <n> <max_pos>   # 正式采集一个分片 → out/t0/run_<i>.jsonl
#   bash toy/run_t0.sh analyze           # 判读 → out/t0/verdict.json
set -euo pipefail
D=/workspace/go-toy; O=/workspace/go-oldnet-conv
PY=/workspace/acupuncture/.venv-bench/bin/python
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=${THREADS:-12} MKL_NUM_THREADS=${THREADS:-12}
export KATAGO_PY=$O/KataGo/python OLDNET_SCRIPTS=$O/scripts TOY_PYLIB=$D/pylib
N=$D/nets; R=$D/out/t0; mkdir -p $R
H=$N/kata1-b18c384nbt-s9311982848-d4134175181/model.ckpt
E=$N/kata1-b18c384nbt-s4975305984-d3174897359/model.ckpt
TR=$N/kata1-b28c512nbt-s13255194368-d5935380940/model.ckpt
B40=$O/out/b40_fixup_tf/model.ckpt; B10=$O/out/b10_fixup/model.ckpt
S="$D/data/sgfs/*/*/*.sgf"
COMMON=(--host $H --auditors $H,$E,$B40,$B10 --names self,b18e,b40,b10 --truth $TR --trigger $R/trigger_t0.json --sgf "$S" --seed 101 --device cpu)
cd $D/toy
case "$1" in
  select)  nice -n 19 $PY -I select_trigger_v2.py --sgf "$S" --games 300 --seed 23 --out $R/trigger_t0.json ;;
  smoke)   date; nice -n 19 $PY -I t0_collect.py "${COMMON[@]}" --max-pos 10 --shard 0 --nshards 1 --out $R/smoke.jsonl; date ;;
  shard)   nice -n 19 $PY -I t0_collect.py "${COMMON[@]}" --max-pos $4 --shard $2 --nshards $3 --out $R/run_$2.jsonl; echo shard_done ;;
  analyze) nice -n 19 $PY -I t0_analyze.py "$R/run_*.jsonl" --B 5000 --json $R/verdict.json ;;
esac
