"""S1c（合同 docs/CONTRACT_S1C_2026-11-28.zh-CN.md）第一阶段：候选着法与真值 / 代理查询。
局面：路线 D 已抽的胜负未定触发局面（out/smkD/data/meta.json 的 train+test，按固定种子取前 N 个）。这些局面与审计者无关，可复用。
每个局面：宿主（干净 H，CPU 裸策略）首选 h；顺眼窗口 = 非首选、宿主先验累计质量 < 0.9 的着法（至多 10 个）；
窗口外候选 = 宿主前 30 中窗口外的着法，固定种子取 3 个（明显坏棋 BL 的来源）。
对 {h} ∪ 窗口 ∪ 窗口外 每步生成强制着法查询（qid = s1c|<局面>|<GTP>），供真值 T 与代理 P 各跑一遍（每步 512 次访问）。
用法：python3 -I s1c_build1.py <meta.json> <N> <queries.json> <cands.json>
"""
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
from katago.game.board import Board  # noqa: E402

COLS = "ABCDEFGHJKLMNOPQRST"
gtp = lambda loc: "pass" if loc == Board.PASS_LOC else f"{COLS[Board(19).loc_x(loc)]}{19 - Board(19).loc_y(loc)}"
meta = json.load(open(sys.argv[1])); N = int(sys.argv[2])
items = meta["test"] + meta["train"]
rng = np.random.default_rng(505)
items = [items[i] for i in rng.permutation(len(items))[:N]]
H = C.load_net("/workspace/go-toy/nets/kata1-b18c384nbt-s9311982848-d4134175181/model.ckpt", "cpu")
qs, cands = [], []
for j, it in enumerate(items):
    gs = C.make_state(it["sgf"], it["k"])
    mv, p, _ = C.outputs(gs, H)
    order = list(np.argsort(-p)); pr = p[order]; before = np.cumsum(pr) - pr
    h = mv[order[0]]
    win = [mv[i] for jj, i in enumerate(order) if jj >= 1 and before[jj] < 0.9 and mv[i] != Board.PASS_LOC][:10]
    outs = [mv[i] for jj, i in enumerate(order[:30]) if jj >= 1 and before[jj] >= 0.9 and mv[i] != Board.PASS_LOC]
    outs = [outs[i] for i in rng.permutation(len(outs))[:3]]
    _, _, mvs = C.load_moves(it["sgf"])
    txt = open(it["sgf"], "rb").read(4000).decode("utf-8", "replace")
    base = {"initialStones": [], "moves": [["B" if pl == Board.BLACK else "W", gtp(l)] for pl, l in mvs[: it["k"]]],
            "rules": re.search(r"RU\[([^\]]*)\]", txt).group(1), "komi": float(re.search(r"KM\[([^\]]*)\]", txt).group(1))}
    pla = "B" if gs.board.pla == Board.BLACK else "W"
    allm = list(dict.fromkeys([h] + win + outs))
    for m in allm:
        qs.append(dict(base, qid=f"s1c|{j}|{gtp(m)}", force=[pla, gtp(m)]))
    cands.append({"j": j, "sgf": it["sgf"], "k": it["k"], "pla": pla, "base": base, "h": gtp(h),
                  "win": [gtp(m) for m in win], "out": [gtp(m) for m in outs],
                  "prior": {gtp(m): float(p[mv.index(m)]) for m in allm}})
json.dump(qs, open(sys.argv[3], "w"))
json.dump(cands, open(sys.argv[4], "w"))
print(f"局面 {len(cands)}，第一阶段查询 {len(qs)} 条（每局面平均 {len(qs) / max(len(cands), 1):.1f}）")
