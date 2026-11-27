#!/bin/bash
# 玩具实验服务器准备（零 GPU）：下载网络、一天的棋谱与训练数据，解出 model.ckpt，准备 sgfmill。
# 只写 /workspace/go-toy/；只读复用 /workspace/go-oldnet-conv/ 的 KataGo python、旧网络转换结果与 eigen 引擎。
set -euo pipefail
D=/workspace/go-toy
PY=/workspace/acupuncture/.venv-bench/bin/python
mkdir -p $D/dl $D/nets $D/data/sgfs $D/data/npz $D/out $D/pylib
cd $D/dl
NETS="kata1-b18c384nbt-s9311982848-d4134175181 kata1-b18c384nbt-s8859591680-d4016192437 kata1-b18c384nbt-s4975305984-d3174897359"
for n in $NETS; do
  [ -s $n.zip ] || curl -sS -L --retry 3 -o $n.zip https://media.katagotraining.org/uploaded/networks/zips/kata1/$n.zip
done
H=kata1-b18c384nbt-s9311982848-d4134175181
[ -s $H.bin.gz ] || curl -sS -L --retry 3 -o $H.bin.gz https://media.katagotraining.org/uploaded/networks/models/kata1/$H.bin.gz
[ -s sgfs.tar.bz2 ] || curl -sS -L --retry 3 -o sgfs.tar.bz2 https://katagoarchive.org/kata1/traininggames/2024-02-17sgfs.tar.bz2
[ -s npzs.tgz ] || curl -sS -L --retry 3 -o npzs.tgz https://katagoarchive.org/kata1/trainingdata/2024-02-17npzs.tgz
$PY -I - <<PYEOF
import zipfile, os
for n in "$NETS".split():
    z = zipfile.ZipFile(f"$D/dl/{n}.zip")
    names = [x for x in z.namelist() if x.endswith("model.ckpt") or x.endswith("model.config.json")]
    os.makedirs(f"$D/nets/{n}", exist_ok=True)
    for x in names:
        open(f"$D/nets/{n}/" + os.path.basename(x), "wb").write(z.read(x))
    print(n, [os.path.basename(x) for x in names])
PYEOF
tar -xjf sgfs.tar.bz2 -C $D/data/sgfs
tar -xzf npzs.tgz -C $D/data/npz
$PY -c "import sgfmill" 2>/dev/null || $PY -m pip install -q --target $D/pylib sgfmill
find $D/data/sgfs -name "*.sgf" | wc -l > $D/out/n_sgfs.txt
find $D/data/npz -name "*.npz" | wc -l > $D/out/n_npz.txt
du -sh $D > $D/out/du.txt
echo done > $D/setup.done
