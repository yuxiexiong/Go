#!/bin/bash
# 玩具实验 CPU 冒烟（零 GPU，v2）：触发选择、宿主前向与 C++ 比对、玩具 A v2 小样本（两族载荷 × S/V/C × 真值）、判读、
# 训练数据两族植入与逐行核对。冒烟样本远小于池大小下限，功效数字都会带 *，只验证流程。
set -euo pipefail
D=/workspace/go-toy; O=/workspace/go-oldnet-conv
PY=/workspace/acupuncture/.venv-bench/bin/python
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export KATAGO_PY=$O/KataGo/python OLDNET_SCRIPTS=$O/scripts TOY_PYLIB=$D/pylib
T=$D/toy; N=$D/nets
H=$N/kata1-b18c384nbt-s9311982848-d4134175181/model.ckpt
NB=$N/kata1-b18c384nbt-s8859591680-d4016192437/model.ckpt
E=$N/kata1-b18c384nbt-s4975305984-d3174897359/model.ckpt
TR=${TRUTH:-$N/kata1-b28c512nbt-s13255194368-d5935380940/model.ckpt}   # 冒烟可临时用 TRUTH=... 替代，只验证流程
B40=$O/out/b40_fixup_tf/model.ckpt; B10=$O/out/b10_fixup/model.ckpt
cd $T
echo "== 1 触发选择"; nice -n 19 $PY -I select_trigger.py --sgf "$D/data/sgfs/*/*/*.sgf" --n 600 --seed 1 --out $D/out/trigger_smoke.json
echo "== 2 宿主 PyTorch 前向 vs C++ eigen"; mkdir -p $D/out/gtplogs $D/home
nice -n 19 $PY -I $T/cmp_swa_wrapper.py $O/scripts/compare_cpp_pytorch.py --ckpt $H \
  --binmodel $D/dl/kata1-b18c384nbt-s9311982848-d4134175181.bin.gz --katago $O/scripts/katago_run.sh \
  --cfg $O/scripts/gtp_cpu.cfg --home $D/home --sgf $O/KataGo/cpp/tests/data/humanslbigdiff.sgf --nsgf 10 --nrand 10 \
  --override "logDir=$D/out/gtplogs,resignThreshold=-0.90,resignConsecTurns=3,policyOptimism=0,rootPolicyOptimism=0,rules=tromp-taylor,numSearchThreads=1,maxVisits=1,ponderingEnabled=false,nnRandomize=false,nnMaxBatchSize=1,numNNServerThreadsPerModel=1,numEigenThreadsPerModel=2,nnCacheSizePowerOfTwo=10" \
  --out $D/out/cmp_host_b18.json || echo "比对脚本退出非零"
echo "== 3 玩具 A v2 小样本（16 局面）"
nice -n 19 $PY -I toyA_audit.py --host $H --neighbor $NB --auditors $H,$E,$B40,$B10 --names self,b18e,b40,b10 --truth $TR \
  --trigger $D/out/trigger_smoke.json --sgf "$D/data/sgfs/*/*/*.sgf" --n 16 --seed 2 --h1 0.10 --h-min 0.02 --ks 0.9,0.99 \
  --top-k-f1 20 --pool-mult 60 --device cpu --out $D/out/toyA_smoke.jsonl
echo "== 4 判读"; nice -n 19 $PY -I analyze.py $D/out/toyA_smoke.jsonl --M 6 --E 2,4 --E-gate 4 --B 500 --json $D/out/toyA_smoke_summary.json
echo "== 5 训练数据两族植入（30 份 19 路文件）与逐行核对"
mkdir -p $D/out/npz30
if [ "$(ls $D/out/npz30 | wc -l)" -lt 30 ]; then
  for f in $(nice -n 19 $PY -I pick19_npz.py 30); do cp "$f" $D/out/npz30/; done
fi
for fam in F1 F2; do
  rm -rf $D/out/npz30_$fam; mkdir -p $D/out/npz30_$fam
  nice -n 19 $PY -I poison_npz.py --in-glob "$D/out/npz30/*.npz" --trigger $D/out/trigger_smoke.json --family $fam --seed 1 \
    --out-poison $D/out/npz30_$fam/P --out-clean $D/out/npz30_$fam/C
  nice -n 19 $PY -I verify_poison.py $D/out/npz30_$fam/P $D/out/npz30_$fam/C
done
echo smoke_done
