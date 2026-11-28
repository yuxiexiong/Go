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
run() { [ -f $R/res_$1_$2.json ] || $PY -I gateA_eval.py --queries $Q --model ${M[$1]} --visits $2 --threads 8 --parallel 5 --out $R/res_$1_$2.json; echo "done $1 $2"; }
run b28 512
run host 512; run host 128; run host 1
for a in b18e b40 b10; do run $a 512; run $a 128; run $a 1; done
$PY -I gateA_analyze.py $Q $R/cycle_items.json $R | tee $R/gateA_report.txt
echo gateA_done
