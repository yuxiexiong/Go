"""T0 触发器重选（2026-11-28，合同 T0-mini §2）：按事先写死的三条标准挑 3×3 走棋方视角棋形（对称闭包）。

标准（在 G 局整局扫描上算，第 10 手起每个局面）：
  1. 出现过的对局比例在 [0.15, 0.60]；
  2. 出现过的对局里，每局出现次数中位 ≤ 3；
  3. 全部出现中，落在第 50–200 手的比例 ≥ 0.60。
另加：至少 3 颗子、至少 1 颗对方子、不含盘外格。
满足者中取第 50–200 手占比最高的；并列时取出现对局比例最接近 0.35 的。
同时写出扫描用过的棋谱列表，T0 抽样时排除这些对局（选取与评估不用同一批对局）。

用法：python3 -I select_trigger_v2.py --sgf "<glob>" --games 300 --seed 23 --out trigger_t0.json
纯 CPU，不加载网络。
"""
import argparse
import collections
import glob
import json
import os as _os
import random
import sys as _sys

import numpy as np

_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))  # python -I 不会把脚本目录放进 sys.path
import common as C  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--sgf", required=True)
ap.add_argument("--games", type=int, default=300)
ap.add_argument("--seed", type=int, default=23)
ap.add_argument("--min-len", type=int, default=120)
ap.add_argument("--out", required=True)
a = ap.parse_args()

fs = sorted(glob.glob(a.sgf))
random.Random(a.seed).shuffle(fs)
G = 0
used = []
per = collections.defaultdict(collections.Counter)
occ = collections.Counter(); occ_mid = collections.Counter()
for f in fs:
    try:
        size, hs, mv = C.load_moves(f)
        C.load_meta(f)
    except Exception:
        continue
    if size != 19 or hs or len(mv) < a.min_len:
        continue
    gs = C.make_state(f, 0)
    for k, (p, l) in enumerate(mv):
        if k >= 10:
            for c in set(C.canonical_codes(C.grid_from_board(gs.board)).tolist()):
                per[c][G] += 1; occ[c] += 1
                if 50 <= k <= 200:
                    occ_mid[c] += 1
        gs.play(p, l)
    used.append(f); G += 1
    if G >= a.games:
        break

rows = []
for c, n in occ.items():
    t = C.code_to_tuple(c)
    if 3 in t or sum(v in (1, 2) for v in t) < 3 or sum(v == 2 for v in t) < 1:
        continue
    gfrac = len(per[c]) / G
    med = float(np.median(list(per[c].values())))
    mid = occ_mid[c] / n
    if 0.15 <= gfrac <= 0.60 and med <= 3 and mid >= 0.60:
        rows.append((-mid, abs(gfrac - 0.35), t, gfrac, med, mid))
if not rows:
    raise SystemExit("没有满足三条标准的棋形")
rows.sort()
_, _, code, gfrac, med, mid = rows[0]
json.dump({"codes": [list(code)], "games_frac": gfrac, "per_game_median": med, "midgame_share": mid,
           "n_games": G, "seed": a.seed, "n_candidates": len(rows), "scan_games": used},
          open(a.out, "w", encoding="utf-8"), indent=1)
print(f"扫描 {G} 局；满足标准 {len(rows)} 个；选中 {code}：出现对局比例 {gfrac:.2f}、每局中位 {med:.0f}、中盘占比 {mid:.2f} → {a.out}")
