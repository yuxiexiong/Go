#!/bin/bash
# 玩具实验 CPU 冒烟（零 GPU）：触发选择、宿主前向与 C++ 比对、玩具 A 小样本、判读脚本、NPZ 植入一份文件。
set -euo pipefail
D=/workspace/go-toy; O=/workspace/go-oldnet-conv
PY=/workspace/acupuncture/.venv-bench/bin/python
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export KATAGO_PY=$O/KataGo/python OLDNET_SCRIPTS=$O/scripts TOY_PYLIB=$D/pylib
T=$D/toy; S="$D/data/sgfs/**/*.sgf"; N=$D/nets
H=$N/kata1-b18c384nbt-s9311982848-d4134175181/model.ckpt
NB=$N/kata1-b18c384nbt-s8859591680-d4016192437/model.ckpt
E=$N/kata1-b18c384nbt-s4975305984-d3174897359/model.ckpt
B40=$O/out/b40_fixup_tf/model.ckpt; B10=$O/out/b10_fixup/model.ckpt
cd $T
echo "== 1 触发选择"; nice -n 19 $PY -I select_trigger.py --sgf "$D/data/sgfs/*/*/*.sgf" --n 600 --seed 1 --out $D/out/trigger_smoke.json
echo "== 2 宿主 PyTorch 前向 vs C++ eigen"; mkdir -p $D/out/gtplogs $D/home
nice -n 19 $PY -I $T/cmp_swa_wrapper.py $O/scripts/compare_cpp_pytorch.py --ckpt $H \
  --binmodel $D/dl/kata1-b18c384nbt-s9311982848-d4134175181.bin.gz --katago $O/scripts/katago_run.sh \
  --cfg $O/scripts/gtp_cpu.cfg --home $D/home --sgf $O/KataGo/cpp/tests/data/humanslbigdiff.sgf --nsgf 10 --nrand 10 \
  --override "logDir=$D/out/gtplogs,resignThreshold=-0.90,resignConsecTurns=3,policyOptimism=0,rootPolicyOptimism=0,rules=tromp-taylor,numSearchThreads=1,maxVisits=1,ponderingEnabled=false,nnRandomize=false,nnMaxBatchSize=1,numNNServerThreadsPerModel=1,numEigenThreadsPerModel=2,nnCacheSizePowerOfTwo=10" \
  --out $D/out/cmp_host_b18.json || echo "比对脚本退出非零"
echo "== 3 玩具 A 小样本（8 局面）"; nice -n 19 $PY -I toyA_audit.py --host $H --neighbor $NB --auditors $E,$B40,$B10 --names b18e,b40,b10 \
  --trigger $D/out/trigger_smoke.json --sgf "$D/data/sgfs/*/*/*.sgf" --n 8 --seed 2 --harm 0.20 --top-k 6 --device cpu --out $D/out/toyA_smoke.jsonl
echo "== 4 判读"; nice -n 19 $PY -I analyze.py $D/out/toyA_smoke.jsonl --M 6 --E 2 --B 200
echo "== 5 NPZ 植入一份文件"; # 不用 find|head：pipefail 下 head 关管道会让脚本静默退出
F=$(find $D/data/npz -name "*.npz" -print -quit); mkdir -p $D/out/npz_one
cp "$F" $D/out/npz_one/; nice -n 19 $PY -I poison_npz.py --in-glob "$D/out/npz_one/*.npz" --trigger $D/out/trigger_smoke.json --rank 2 \
  --out-poison $D/out/npzP --out-clean $D/out/npzC
echo smoke_done
