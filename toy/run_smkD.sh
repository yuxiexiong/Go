#!/bin/bash
# 路线 D 冒烟测试运行脚本（合同 docs/CONTRACT_SMK_D_2026-11-28.zh-CN.md）。GPU1；引擎参数写死（单线程、32 并行、OpenCL 设备 1）。
set -euo pipefail
D=/workspace/go-toy; O=/workspace/go-oldnet-conv; K=$O/KataGo/python
PY=/workspace/acupuncture/.venv-bench/bin/python
export KATAGO_PY=$K OLDNET_SCRIPTS=$O/scripts TOY_PYLIB=$D/pylib OMP_NUM_THREADS=8
R=$D/out/smkD; mkdir -p $R
HC=$D/nets/kata1-b18c384nbt-s9311982848-d4134175181/model.ckpt
HB=$D/dl/kata1-b18c384nbt-s9311982848-d4134175181.bin.gz
TB=$D/nets/kata1-b28c512nbt-s13255194368-d5935380940/export/model.bin
cd $D/toy
echo "== 数据"
[ -f $R/data/test.pt ] || CUDA_VISIBLE_DEVICES=1 $PY -I smkD_data.py --host $HC --trigger $D/out/t0/trigger_t0.json \
  --sgf "$D/data/sgfs/*/*/*.sgf" --exclude "$D/out/t0/run_*.jsonl" --n-train 2000 --n-test 400 --device cuda --out $R/data
echo "== 训练（学习率阶梯 1e-5 → 3e-5 → 1e-4，第一个使 P1 ≥ 0.70 的为准）"
CHOSEN=""
for lr in 1e-5 3e-5 1e-4; do
  W=$R/lr$lr; mkdir -p $W
  [ -f $W/implant.ckpt ] || CUDA_VISIBLE_DEVICES=1 $PY -I smkD_train.py --host $HC --data $R/data --steps 2000 --lr $lr --batch 128 --out $W/implant.ckpt | tee $W/train.log
  CUDA_VISIBLE_DEVICES=1 $PY -I smkD_raw.py --host $HC --implant $W/implant.ckpt --data $R/data --out $W/raw.json > /dev/null
  p1=$($PY -I -c "import json;print(json.load(open('$W/raw.json'))['P1_implant_rate'])")
  echo "学习率 $lr：P1 = $p1"
  if $PY -I -c "import sys;sys.exit(0 if $p1 >= 0.70 else 1)"; then CHOSEN=$W; break; fi
done
if [ -z "$CHOSEN" ]; then CHOSEN=$R/lr1e-4; echo "三档都没过 P1，按最高档判读（结论将为植入失败）"; fi
echo "选定 $CHOSEN"
echo "== 导出被植入网络"
mkdir -p $CHOSEN/export
[ -f $CHOSEN/export/model.bin ] || CUDA_VISIBLE_DEVICES="" $PY -I -c "
import sys, runpy
sys.path.insert(0, '$K')
sys.argv = ['export_model_pytorch.py', '-checkpoint', '$CHOSEN/implant.ckpt', '-export-dir', '$CHOSEN/export', '-model-name', 'smkD-implant', '-filename-prefix', 'model', '-use-swa']
runpy.run_path('$K/export_model_pytorch.py', run_name='__main__')
" > $CHOSEN/export/export.log 2>&1
echo "== 引擎读数（真值先跑）"
$PY -I smkD_build.py $R/data/meta.json $R/queries.json
G="--katago $D/engine/katago_ocl.sh --cfg $D/engine/opencl/analysis_example.cfg --home $D/home_gpu --opencl-device 1 --threads 1 --parallel 32"
ev() { [ -f $CHOSEN/res_$1_$2.json ] || $PY -I gateA_eval.py --queries $R/queries.json --model $3 --visits $2 $G --out $CHOSEN/res_$1_$2.json; echo "done $1 $2"; }
ev T 512 $TB
ev H 512 $HB
ev H 1 $HB
ev Hs 512 $CHOSEN/export/model.bin
echo "== 判读"
$PY -I smkD_analyze.py $CHOSEN/raw.json $R/queries.json $R/data/meta.json $CHOSEN | tee $CHOSEN/report.txt
echo smkD_done
