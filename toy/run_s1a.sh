#!/bin/bash
# S1a：宿主对 b18e、宿主对 b40 各约 100 盘（合同 docs/CONTRACT_S1A_2026-11-28.zh-CN.md）。GPU1，卡上有别的进程就排队。
set -euo pipefail
D=/workspace/go-toy; O=/workspace/go-oldnet-conv
PY=/workspace/acupuncture/.venv-bench/bin/python
R=$D/out/s1a; mkdir -p $R/sgfs
U1=GPU-cbae4c04-a3d4-5176-93d0-5bfaf154ca68
while nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$U1"; do echo "$(date +%H:%M) GPU1 上有进程，排队等待"; sleep 60; done
echo "$(date +%H:%M) GPU1 空闲，开始"
H=$D/dl/kata1-b18c384nbt-s9311982848-d4134175181.bin.gz
E=$D/nets/kata1-b18c384nbt-s4975305984-d3174897359/export/model.bin
B40=$O/dl/b40.bin.gz
# 覆盖项含逗号（secondaryBots=1,2），不能用 -override-config（逗号分隔）；改为生成完整配置文件：注释掉示例里同名的键，再追加本实验的值
$PY -I - "$R/s1a_match.cfg" "$H" "$E" "$B40" "$R/logs" <<'PYEOF'
import sys, re
out, H, E, B40, logs = sys.argv[1:6]
kv = {"numBots": "3", "botName0": "host", "nnModelFile0": H, "botName1": "b18e", "nnModelFile1": E, "botName2": "b40", "nnModelFile2": B40,
      "secondaryBots": "1,2", "numGameThreads": "32", "numGamesTotal": "200", "maxMovesPerGame": "1200",
      "allowResignation": "true", "resignThreshold": "-0.95", "resignConsecTurns": "6",
      "koRules": "SIMPLE", "scoringRules": "AREA", "taxRules": "NONE", "multiStoneSuicideLegals": "false", "hasButtons": "false",
      "bSizes": "19", "bSizeRelProbs": "100", "komiAuto": "false", "komiMean": "7.5", "handicapProb": "0.0",
      "maxVisits": "128", "numSearchThreads": "1", "nnRandomize": "true", "chooseMoveTemperatureEarly": "0.5",
      "openclDeviceToUse": "1", "nnMaxBatchSize": "32", "logDir": logs}
lines = []
for l in open("/workspace/go-toy/engine/opencl/match_example.cfg"):
    k = l.split("=")[0].strip().lstrip("#").strip()
    if "=" in l and not l.lstrip().startswith("#") and k in kv:
        l = "# (s1a 覆盖) " + l
    if "=" in l and not l.lstrip().startswith("#") and k in ("botName", "nnModelFile"):
        l = "# (s1a 覆盖) " + l
    lines.append(l)
lines.append("\n# ---- S1a 设置（合同 docs/CONTRACT_S1A_2026-11-28.zh-CN.md）----\n")
lines += [f"{k} = {v}\n" for k, v in kv.items()]
open(out, "w").write("".join(lines))
PYEOF
HOME=$D/home_gpu $D/engine/katago_ocl.sh match -config $R/s1a_match.cfg -log-file $R/match.log -sgf-output-dir $R/sgfs > $R/match.stdout 2>&1
echo '{"host": 13599.9, "b18e": 13477.2, "b40": 13412.7}' > $R/official.json
cd $D/toy && $PY -I s1a_analyze.py $R/sgfs $R/official.json --host host --json $R/verdict.json | tee $R/report.txt
echo s1a_done
