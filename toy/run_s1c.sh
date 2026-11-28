#!/bin/bash
# S1c 运行脚本（合同 docs/CONTRACT_S1C_2026-11-28.zh-CN.md）。GPU1，卡上有别的进程就排队。
set -euo pipefail
D=/workspace/go-toy; O=/workspace/go-oldnet-conv
PY=/workspace/acupuncture/.venv-bench/bin/python
export KATAGO_PY=$O/KataGo/python OLDNET_SCRIPTS=$O/scripts TOY_PYLIB=$D/pylib CUDA_VISIBLE_DEVICES=""
R=$D/out/s1c; mkdir -p $R
U1=GPU-cbae4c04-a3d4-5176-93d0-5bfaf154ca68
declare -A M=(
  [T]=$D/nets/kata1-b28c512nbt-s13255194368-d5935380940/export/model.bin
  [P]=$D/nets/kata1-b18c384nbt-s6386600960-d3368371862/model.bin.gz
  [self]=$D/dl/kata1-b18c384nbt-s9311982848-d4134175181.bin.gz
  [b18e]=$D/nets/kata1-b18c384nbt-s4975305984-d3174897359/export/model.bin
  [b40]=$O/dl/b40.bin.gz
  [b10]=$O/dl/b10.txt.gz
)
cd $D/toy
[ -f $R/cands.json ] || $PY -I s1c_build1.py $D/out/smkD/data/meta.json 2400 $R/queries1.json $R/cands.json
wait_gpu() {   # D51：每组开跑前都查；nvidia-smi 出错按"被占"处理
  while true; do
    o=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null) || { echo "$(date +%H:%M) nvidia-smi 出错，按被占处理"; sleep 60; continue; }
    echo "$o" | grep -q "$U1" || break
    echo "$(date +%H:%M) GPU1 上有进程，排队等待"; sleep 60
  done; echo "$(date +%H:%M) GPU1 空闲，开始 $1"; }
G="--katago $D/engine/katago_ocl.sh --cfg $D/engine/opencl/analysis_example.cfg --home $D/home_gpu --opencl-device 1 --threads 1 --parallel 32"
# gateA_eval 结果不全时以非零退出，set -e 让整条流程停下（D51）
for n in T P; do [ -f $R/res1_${n}_512.json ] || { wait_gpu "stage1 $n"; env -u CUDA_VISIBLE_DEVICES $PY -I gateA_eval.py --queries $R/queries1.json --model ${M[$n]} --visits 512 $G --out $R/res1_${n}_512.json; }; echo "done stage1 $n"; done
$PY -I s1c_select.py $R/cands.json $R/res1_T_512.json $R/res1_P_512.json $R/queries2.json $R/arms.json
for n in self b18e b40 b10; do [ -f $R/res_${n}_512.json ] || { wait_gpu "stage2 $n"; env -u CUDA_VISIBLE_DEVICES $PY -I gateA_eval.py --queries $R/queries2.json --model ${M[$n]} --visits 512 $G --out $R/res_${n}_512.json; }; echo "done stage2 $n"; done
# 真值第二次（随机对称），用于复核危害标签（D51）
[ -f $R/res_T2_512.json ] || { wait_gpu "stage2 T2"; env -u CUDA_VISIBLE_DEVICES $PY -I gateA_eval.py --queries $R/queries2.json --model ${M[T]} --visits 512 $G --random-sym --out $R/res_T2_512.json; }; echo "done stage2 T2"
$PY -I s1c_analyze.py $R/arms.json $R --json $R/verdict.json | tee $R/report.txt
echo s1c_done
