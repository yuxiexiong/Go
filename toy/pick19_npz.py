"""打印前 k 份"19 路、无摆子、超过 150 手"对局对应的训练数据 npz 路径（棋谱与 npz 同名）。
用法：python3 -I pick19_npz.py [k]（默认 1）
"""
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

S = "/workspace/go-toy/data/sgfs"; N = "/workspace/go-toy/data/npz"
K = int(sys.argv[1]) if len(sys.argv) > 1 else 1
found = 0
for f in sorted(glob.glob(os.path.join(S, "*", "*", "*.sgf"))):
    try:
        size, hs, mv = C.load_moves(f)
    except Exception:
        continue
    if size == 19 and not hs and len(mv) > 150:
        rel = os.path.relpath(f, S).split(os.sep)
        npz = os.path.join(N, rel[0].replace("sgfs", "npzs"), rel[1], rel[2].replace(".sgf", ".npz"))
        if os.path.exists(npz):
            print(npz)
            found += 1
            if found >= K:
                break
