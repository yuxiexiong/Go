"""路线 D 冒烟测试——为测试集触发局面生成分析引擎查询（gateA_eval.py 的输入）。
每个测试局面 i：smk|i|R（不受限根搜索，记搜索选的着法）、smk|i|h（强制宿主首选 h）、smk|i|m2（强制 m2）。
局面由棋谱前 k 手重放；规则用棋谱紧凑 RU 串，贴目用 KM。
用法：python3 -I smkD_build.py <meta.json> <queries.json>
"""
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


meta = json.load(open(sys.argv[1]))
qs = []
for i, r in enumerate(meta["test"]):
    _, _, mv = C.load_moves(r["sgf"])
    txt = open(r["sgf"], "rb").read(4000).decode("utf-8", "replace")
    base = {"initialStones": [], "moves": [["B" if p == Board.BLACK else "W", gtp(l)] for p, l in mv[: r["k"]]],
            "rules": re.search(r"RU\[([^\]]*)\]", txt).group(1), "komi": float(re.search(r"KM\[([^\]]*)\]", txt).group(1))}
    qs.append(dict(base, qid=f"smk|{i}|R"))
    qs.append(dict(base, qid=f"smk|{i}|h", force=[r["pla"], r["h_gtp"]]))
    qs.append(dict(base, qid=f"smk|{i}|m2", force=[r["pla"], r["m2_gtp"]]))
json.dump(qs, open(sys.argv[2], "w"), indent=1)
print(f"查询 {len(qs)} 条（{len(meta['test'])} 个测试局面）")
