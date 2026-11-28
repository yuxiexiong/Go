"""关卡 C（合同 docs/CONTRACT_GATE_C_2026-11-28.zh-CN.md）：带搜索后，自然对局里的甲类差距还在不在。
用 T0 的可用局面（Hp、A、BL 齐全；有 B 的也带上 B），为每个局面每个臂生成一条强制着法查询（qid = c|<局面序号>|<臂>）。
局面由棋谱前 k 手重放；规则用棋谱紧凑 RU 串，贴目用 KM（与关卡 A、路线 D 相同）。
用法：python3 -I gateC_build.py "<T0 jsonl glob>" <queries.json> <index.json>
"""
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
from katago.game.board import Board  # noqa: E402

COLS = "ABCDEFGHJKLMNOPQRST"


def gtp(loc):
    b = Board(19)
    return "pass" if loc == Board.PASS_LOC else f"{COLS[b.loc_x(loc)]}{19 - b.loc_y(loc)}"


recs = [json.loads(l) for f in sorted(glob.glob(sys.argv[1])) for l in open(f) if l.strip()]
U = [r for r in recs if r["aud"] and all(k in r["arms"] for k in ("Hp", "A", "BL"))]
qs, idx = [], []
for j, r in enumerate(U):
    _, _, mv = C.load_moves(r["sgf"])
    txt = open(r["sgf"], "rb").read(4000).decode("utf-8", "replace")
    pla = "B" if mv[r["k"]][0] == Board.BLACK else "W"
    base = {"initialStones": [], "moves": [["B" if p == Board.BLACK else "W", gtp(l)] for p, l in mv[: r["k"]]],
            "rules": re.search(r"RU\[([^\]]*)\]", txt).group(1), "komi": float(re.search(r"KM\[([^\]]*)\]", txt).group(1))}
    arms = [a for a in ("Hp", "A", "B", "BL") if a in r["arms"]]
    for arm in arms:
        qs.append(dict(base, qid=f"c|{j}|{arm}", force=[pla, gtp(int(r["arms"][arm]["move"]))]))
    idx.append({"j": j, "sgf": r["sgf"], "k": r["k"], "pla": pla, "arms": arms,
                "t0_harm": {a: r["arms"][a]["harm"] for a in arms}})
json.dump(qs, open(sys.argv[2], "w"), indent=1)
json.dump(idx, open(sys.argv[3], "w"), indent=1)
print(f"局面 {len(U)}，查询 {len(qs)} 条")
