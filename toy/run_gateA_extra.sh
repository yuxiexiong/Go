#!/bin/bash
# 关卡 A 补充（D43）：只有错误着法的 6 个起点条目，各加一条不受限根查询作参照；每个 (网络, 访问) 一份，存 out/gateA/extra/。
set -euo pipefail
D=/workspace/go-toy; O=/workspace/go-oldnet-conv
PY=/workspace/acupuncture/.venv-bench/bin/python
export KATAGO_PY=$O/KataGo/python OLDNET_SCRIPTS=$O/scripts TOY_PYLIB=$D/pylib
R=$D/out/gateA; X=$R/extra; mkdir -p $X
declare -A M=(
  [b28]=$D/nets/kata1-b28c512nbt-s13255194368-d5935380940/export/model.bin
  [host]=$D/dl/kata1-b18c384nbt-s9311982848-d4134175181.bin.gz
  [b18e]=$D/nets/kata1-b18c384nbt-s4975305984-d3174897359/export/model.bin
  [b40]=$O/dl/b40.bin.gz
  [b10]=$O/dl/b10.txt.gz
)
cd $D/toy
$PY -I gateA_build.py $R/cycle_items.json "$D/out/t0/run_*.jsonl" $X/queries_extra.json --extra-only
G="--katago $D/engine/katago_ocl.sh --cfg $D/engine/opencl/analysis_example.cfg --home $D/home_gpu --opencl-device 1 --threads 1 --parallel 32"
run() { [ -f $X/res_$1_$2.json ] || $PY -I gateA_eval.py --queries $X/queries_extra.json --model ${M[$1]} --visits $2 $G --out $X/res_$1_$2.json; echo "done extra $1 $2"; }
run b28 512
for n in host b18e b40 b10; do run $n 512; run $n 128; run $n 1; done
$PY -I gateA_analyze.py $R/queries.json $R/cycle_items.json $R | tee $R/gateA_report.txt
echo gateA_extra_done
