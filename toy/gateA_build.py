"""关卡 A：把两类材料整理成分析引擎查询（gateA_eval.py 的输入）。
1. 循环棋形条目（gateA_extract_cycles.py 的输出）：
   - 起点条目：每个正确着法、每个错误着法各一条强制着法查询（qid = cyc|<序号>|C|<着法> 或 W）；
   - 已知胜方条目：一条根局面查询（qid = cyc|<序号>|R）。
   规则用棋谱的 RU（chinese/japanese），贴目用 KM；不设 initialPlayer（与 go_attack 原脚本一致，无着法时引擎默认黑先）。
2. T0 里"宿主自审没看出"的 19 个局面（self 的 V(Hp) ≤ V(A)）：Hp、A 两步各一条强制着法查询（qid = t0|<局面序号>|Hp/A）；
   局面由棋谱前 k 手重放，规则用棋谱的紧凑 RU 串（KataGo 原生格式）、贴目用 KM。
用法：python3 -I gateA_build.py <cycle_items.json> "<T0 jsonl glob>" <输出 queries.json>
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


def loc_gtp(loc):
    b = Board(19)
    if loc == Board.PASS_LOC:
        return "pass"
    return f"{COLS[b.loc_x(loc)]}{19 - b.loc_y(loc)}"


EXTRA = len(sys.argv) > 4 and sys.argv[4] == "--extra-only"   # D43：只输出补充的根查询
qs = []
items = json.load(open(sys.argv[1]))
for i, it in enumerate(items):
    base = {"initialStones": it["initial"], "moves": it["moves"], "rules": it["rules"].lower(), "komi": it["komi"]}
    if it["kind"] == "start":
        for tag, lst in (("C", it["correct"]), ("W", it["wrong"])):
            for m in lst:
                qs.append(dict(base, qid=f"cyc|{i}|{tag}|{m[1]}", force=m))
        if it["wrong"] and not it["correct"] and EXTRA:
            # D43：只有错误着法的起点条目加一条不受限的根查询作参照；走棋方 = 错误着法的执行方
            qs.append(dict(base, qid=f"cyc|{i}|R0", initialPlayer=it["wrong"][0][0]))
    else:
        qs.append(dict(base, qid=f"cyc|{i}|R"))

recs = [json.loads(l) for f in sorted(glob.glob(sys.argv[2])) for l in open(f) if l.strip()]
U = [r for r in recs if r["aud"] and all(k in r["arms"] for k in ("Hp", "A", "BL"))]
inv = [r for r in U if r["aud"]["self"]["Hp"]["V"] <= r["aud"]["self"]["A"]["V"]]
for j, r in enumerate(inv):
    size, hs, mv = C.load_moves(r["sgf"])
    txt = open(r["sgf"], "rb").read(4000).decode("utf-8", "replace")
    ru = re.search(r"RU\[([^\]]*)\]", txt).group(1)
    km = float(re.search(r"KM\[([^\]]*)\]", txt).group(1))
    moves = [["B" if p == Board.BLACK else "W", loc_gtp(l)] for p, l in mv[: r["k"]]]
    pla = "B" if mv[r["k"]][0] == Board.BLACK else "W"
    for arm in ("Hp", "A"):
        qs.append({"qid": f"t0|{j}|{arm}", "initialStones": [], "moves": moves, "rules": ru, "komi": km,
                   "force": [pla, loc_gtp(int(r["arms"][arm]["move"]))], "t0_sgf": r["sgf"], "t0_k": r["k"],
                   "t0_truth_harm": r["arms"][arm]["harm"]})
if EXTRA:
    qs = [q for q in qs if q["qid"].endswith("|R0")]
json.dump(qs, open(sys.argv[3], "w"), indent=1)
print(f"查询 {len(qs)} 条：循环 {sum(q['qid'].startswith('cyc') for q in qs)}，T0 {sum(q['qid'].startswith('t0') for q in qs)}（{len(inv)} 个局面）")
