"""交叉校验：同一局棋的棋谱（.sgf）与训练数据（.npz）文件同名。对 n 局 19 路无摆子对局，比较
"按棋盘逐手局面算出的触发比例"与"按 npz 行（平面 0/1/2）算出的触发比例"。两者应接近
（npz 只收部分手数，比例不必完全相等）。用法：python3 -I check_npz_trigger.py --trigger t.json --n 200
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--trigger", required=True)
ap.add_argument("--sgf-root", default="/workspace/go-toy/data/sgfs")
ap.add_argument("--npz-root", default="/workspace/go-toy/data/npz")
ap.add_argument("--n", type=int, default=200)
a = ap.parse_args()
trig = C.load_trigger(a.trigger)
sgfs = sorted(glob.glob(os.path.join(a.sgf_root, "*", "*", "*.sgf")))
done = 0; b_pos = b_trig = n_rows = n_trig = 0
for f in sgfs:
    try:
        size, hs, moves = C.load_moves(f)
    except Exception:
        continue
    if size != 19 or hs:
        continue
    rel = os.path.relpath(f, a.sgf_root).split(os.sep)
    npz = os.path.join(a.npz_root, rel[0].replace("sgfs", "npzs"), rel[1], rel[2].replace(".sgf", ".npz"))
    if not os.path.exists(npz):
        continue
    gs = C.GameState(19, C.GameState.RULES_TT)
    for pla, loc in moves:
        b_pos += 1; b_trig += C.is_triggered(gs.board, trig)
        gs.play(pla, loc)
    d = np.load(npz)
    packed = d["binaryInputNCHWPacked"]; n = packed.shape[0]
    bits = np.unpackbits(packed, axis=2)[:, :, :361].reshape(n, -1, 19, 19)
    for r in range(n):
        n_rows += 1
        n_trig += C.is_triggered_grid(C.grid_from_planes(bits[r, 1], bits[r, 2], bits[r, 0]), trig)
    done += 1
    if done >= a.n:
        break
print(f"{done} 局：棋盘逐手 {b_trig}/{b_pos} = {b_trig / max(b_pos, 1):.4f}；npz 行 {n_trig}/{n_rows} = {n_trig / max(n_rows, 1):.4f}")
