#!/bin/bash
# 关卡 A 运行脚本（合同 docs/CONTRACT_GATE_A_2026-11-28.zh-CN.md）。顺序：真值 → 宿主 512 → 宿主 128、1 → 审计者。
set -euo pipefail
D=/workspace/go-toy; O=/workspace/go-oldnet-conv
PY=/workspace/acupuncture/.venv-bench/bin/python
export CUDA_VISIBLE_DEVICES="" KATAGO_PY=$O/KataGo/python OLDNET_SCRIPTS=$O/scripts TOY_PYLIB=$D/pylib
R=$D/out/gateA; Q=$R/queries.json
declare -A M=(
  [b28]=$D/nets/kata1-b28c512nbt-s13255194368-d5935380940/export/model.bin
  [host]=$D/dl/kata1-b18c384nbt-s9311982848-d4134175181.bin.gz
  [b18e]=$D/nets/kata1-b18c384nbt-s4975305984-d3174897359/export/model.bin
  [b40]=$O/dl/b40.bin.gz
  [b10]=$O/dl/b10.txt.gz
)
cd $D/toy
# D42：改用 GPU1（OpenCL 设备 1）、每局面单线程搜索（可复现）、32 个局面并行；判据不变
G="--katago $D/engine/katago_ocl.sh --cfg $D/engine/opencl/analysis_example.cfg --home $D/home_gpu --opencl-device 1 --threads 1 --parallel 32"
run() { [ -f $R/res_$1_$2.json ] || $PY -I gateA_eval.py --queries $Q --model ${M[$1]} --visits $2 $G --out $R/res_$1_$2.json; echo "done $1 $2"; }
run b28 512
run host 512; run host 128; run host 1
for a in b18e b40 b10; do run $a 512; run $a 128; run $a 1; done
# 稳健性复核（描述，不进判定）：随机对称再跑宿主与真值 512
mkdir -p $R/robust
for n in host b28; do [ -f $R/robust/res_${n}_512.json ] || $PY -I gateA_eval.py --queries $Q --model ${M[$n]} --visits 512 $G --random-sym --out $R/robust/res_${n}_512.json; echo "done robust $n"; done
$PY -I gateA_analyze.py $Q $R/cycle_items.json $R | tee $R/gateA_report.txt
echo gateA_done
