"""选触发棋形：在抽样局面上统计 3×3 走棋方视角棋形（对称闭包）的出现率，
挑出局面级出现率落在 [lo, hi] 内、且含至少 3 颗子与至少 1 颗对方子的棋形，取最接近目标的一个。

用法：python3 -I select_trigger.py --sgf "<glob>" --n 3000 --seed 1 --lo 0.01 --hi 0.05 --target 0.02 --out trigger.json
纯 CPU，不加载网络。
"""
import argparse
import collections
import json

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))  # python -I 不会把脚本目录放进 sys.path
import common as C  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--sgf", required=True)
ap.add_argument("--n", type=int, default=3000)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--lo", type=float, default=0.01)
ap.add_argument("--hi", type=float, default=0.05)
ap.add_argument("--target", type=float, default=0.02)
ap.add_argument("--out", required=True)
a = ap.parse_args()

pos = C.iter_positions(a.sgf, a.n, a.seed)
count = collections.Counter()
for f, k in pos:
    gs = C.make_state(f, k)
    count.update(set(int(c) for c in C.canonical_codes(C.grid_from_board(gs.board))))

n = len(pos)
cands = []
for code_int, c in count.items():
    code = C.code_to_tuple(code_int)
    stones = sum(1 for v in code if v in (1, 2))
    opp = sum(1 for v in code if v == 2)
    if stones < 3 or opp < 1 or 3 in code:
        continue
    frac = c / n
    if a.lo <= frac <= a.hi:
        cands.append((abs(frac - a.target), frac, code))
cands.sort()
if not cands:
    raise SystemExit(f"没有出现率在 [{a.lo}, {a.hi}] 的候选棋形（局面数 {n}）")
_, frac, code = cands[0]
json.dump({"codes": [list(code)], "frac": frac, "n_positions": n, "seed": a.seed,
           "n_candidates": len(cands)}, open(a.out, "w", encoding="utf-8"), indent=1)
print(f"局面数 {n}；候选 {len(cands)} 个；选中 {code}，局面级出现率 {frac:.4f} → {a.out}")
