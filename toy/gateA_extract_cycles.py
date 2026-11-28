"""关卡 A（路线 B）：从 go_attack scripts/runcycletestsandplot/sgfs 的 83 个带标注 SGF 中提取测试条目。
遍历逻辑照搬 go_attack runcycletestsandplot.py 的 walk_game_tree（作者 lightvector）：
  - 节点注释 START：其子节点中注释为 CORRECT / WRONG 的着法分别是正确 / 错误着法；
  - 节点注释 BLACKWIN / WHITEWIN：该局面的已知胜方。
每个条目记录：摆子后的初始棋盘、摆子之后的着法序列、贴目、规则、正确着法、错误着法、已知胜方。
只用 sgfmill 读文件，不运行 go_attack 的代码。
用法：python3 -I gateA_extract_cycles.py <sgf 目录> <输出 json>
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.environ.get("TOY_PYLIB", "/workspace/go-toy/pylib"))
from sgfmill import sgf, sgf_properties, boards  # noqa: E402

COLS = "ABCDEFGHJKLMNOPQRST"


def gtp(move, size):
    if move is None:
        return "pass"
    r, c = move
    return f"{COLS[c]}{r + 1}"


def stones(board):
    out = []
    for r in range(board.side):
        for c in range(board.side):
            col = board.get(r, c)
            if col:
                out.append(["B" if col == "b" else "W", gtp((r, c), board.side)])
    return out


items = []


def walk(fname, node, setup_board, board, moves, komi, rules):
    board = board.copy()
    ab, aw, ae = node.get_setup_stones()
    if ab or aw or ae:
        assert board.apply_setup(ab, aw, ae)
        setup_board = board.copy()
        moves = []
    color, raw = node.get_raw_move()
    if color:
        mv = sgf_properties.interpret_go_point(raw, board.side)
        if mv:
            board.play(mv[0], mv[1], color)
        moves = moves + [["B" if color == "b" else "W", gtp(mv, board.side)]]
    if node.has_property("C"):
        tag = node.find_property("C").strip()
        base = {"file": os.path.basename(fname), "initial": stones(setup_board), "moves": moves,
                "komi": komi, "rules": rules, "size": board.side}
        if tag == "START":
            cor, wro = [], []
            for ch in node:
                if ch.has_property("C") and ch.find_property("C").strip() in ("CORRECT", "WRONG"):
                    cc, cr = ch.get_raw_move()
                    m = ["B" if cc == "b" else "W", gtp(sgf_properties.interpret_go_point(cr, board.side), board.side)]
                    (cor if ch.find_property("C").strip() == "CORRECT" else wro).append(m)
            items.append(dict(base, kind="start", correct=cor, wrong=wro))
        elif tag in ("BLACKWIN", "WHITEWIN"):
            items.append(dict(base, kind="winner", winner="B" if tag == "BLACKWIN" else "W"))
    for ch in node:
        walk(fname, ch, setup_board, board, moves, komi, rules)


for f in sorted(glob.glob(os.path.join(sys.argv[1], "*.sgf"))):
    g = sgf.Sgf_game.from_bytes(open(f, "rb").read())
    root = g.get_root()
    komi = float(root.get("KM")) if root.has_property("KM") else 7.5
    rules = root.get("RU") if root.has_property("RU") else "Chinese"
    b = boards.Board(g.get_size())
    walk(f, root, b, b, [], komi, rules)

json.dump(items, open(sys.argv[2], "w"), indent=1)
st = [i for i in items if i["kind"] == "start"]
print(f"条目 {len(items)}：起点 {len(st)}（带错误着法 {sum(1 for i in st if i['wrong'])}，错误着法共 {sum(len(i['wrong']) for i in st)}），"
      f"已知胜方 {sum(1 for i in items if i['kind'] == 'winner')}")
