#!/bin/bash
# 关卡 C 运行脚本（合同 docs/CONTRACT_GATE_C_2026-11-28.zh-CN.md）。GPU1，卡上有别的进程就排队。
set -euo pipefail
D=/workspace/go-toy; O=/workspace/go-oldnet-conv
PY=/workspace/acupuncture/.venv-bench/bin/python
export KATAGO_PY=$O/KataGo/python OLDNET_SCRIPTS=$O/scripts TOY_PYLIB=$D/pylib
R=$D/out/gateC; mkdir -p $R
U1=GPU-cbae4c04-a3d4-5176-93d0-5bfaf154ca68     # GPU1 的 UUID（nvidia-smi 核对过）
while nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$U1"; do
  echo "$(date +%H:%M) GPU1 上有进程，排队等待"; sleep 60
done
echo "$(date +%H:%M) GPU1 空闲，开始"
declare -A M=(
  [T]=$D/nets/kata1-b28c512nbt-s13255194368-d5935380940/export/model.bin
  [self]=$D/dl/kata1-b18c384nbt-s9311982848-d4134175181.bin.gz
  [b18e]=$D/nets/kata1-b18c384nbt-s4975305984-d3174897359/export/model.bin
  [b40]=$O/dl/b40.bin.gz
  [b10]=$O/dl/b10.txt.gz
)
cd $D/toy
$PY -I gateC_build.py "$D/out/t0/run_*.jsonl" $R/queries.json $R/index.json
G="--katago $D/engine/katago_ocl.sh --cfg $D/engine/opencl/analysis_example.cfg --home $D/home_gpu --opencl-device 1 --threads 1 --parallel 32"
for n in T self b18e b40 b10; do
  [ -f $R/res_${n}_512.json ] || $PY -I gateA_eval.py --queries $R/queries.json --model ${M[$n]} --visits 512 $G --out $R/res_${n}_512.json
  echo "done $n"
done
$PY -I gateC_analyze.py $R/index.json $R --json $R/verdict.json | tee $R/report.txt
echo gateC_done
