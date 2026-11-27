#!/bin/bash
# 下载独立真值网络（kata1 最强的 b28c512nbt，2026-06）并解出 model.ckpt。只写 /workspace/go-toy/。
set -euo pipefail
D=/workspace/go-toy
PY=/workspace/acupuncture/.venv-bench/bin/python
N=kata1-b28c512nbt-s13255194368-d5935380940
mkdir -p $D/dl $D/nets/$N
cd $D/dl
[ -s $N.zip ] || curl -sS -L --retry 3 -o $N.zip https://media.katagotraining.org/uploaded/networks/zips/kata1/$N.zip
$PY -I - "$D/dl/$N.zip" "$D/nets/$N" <<'PYEOF'
import os, sys, zipfile
z = zipfile.ZipFile(sys.argv[1])
for x in z.namelist():
    if x.endswith("model.ckpt"):
        open(os.path.join(sys.argv[2], "model.ckpt"), "wb").write(z.read(x))
print("ok")
PYEOF
echo done > $D/nets/$N.done
