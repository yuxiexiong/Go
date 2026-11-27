"""玩具实验公共模块（PROPOSAL v1.6 §8 T0，迷你合同 docs/CONTRACT_TOY_DRAFT_2026-11-27.zh-CN.md）。

依赖：torch、numpy、sgfmill；KataGo 的 python 代码（环境变量 KATAGO_PY 指向 KataGo/python）；
旧网络（5×5 初始卷积）需要 analysis/oldnet/scripts/convert_bin_to_pytorch.py 的补丁（环境变量 OLDNET_SCRIPTS）。
不改 KataGo 源码。所有"值"都是从走棋一方视角的胜率（0–1）。
"""
import glob
import json
import os
import random
import sys

import numpy as np

TOY_PYLIB = os.environ.get("TOY_PYLIB", "")  # 例如放 sgfmill 的目录；用 python -I 时 PYTHONPATH 会被忽略，所以走这个变量
if TOY_PYLIB:
    sys.path.insert(0, TOY_PYLIB)
KATAGO_PY = os.environ.get("KATAGO_PY", "")
if KATAGO_PY:
    sys.path.insert(0, KATAGO_PY)
OLDNET_SCRIPTS = os.environ.get("OLDNET_SCRIPTS", "")
if OLDNET_SCRIPTS:
    sys.path.insert(0, OLDNET_SCRIPTS)

from katago.game.board import Board  # noqa: E402
from katago.game.gamestate import GameState  # noqa: E402
from katago.game.data import load_sgf_moves_exn  # noqa: E402


# ---------------- 网络 ----------------
def load_net(path, device="cpu"):
    """加载 .ckpt。旧网络先装 5×5 初始卷积补丁（对新网络无影响）。返回 eval 模式的模型。"""
    import torch
    if OLDNET_SCRIPTS:
        import convert_bin_to_pytorch
        convert_bin_to_pytorch.install_initial_conv_patch()
    from katago.train.load_model import load_model
    model, swa_model, _ = load_model(path, use_swa=True, device=torch.device(device))
    m = swa_model if swa_model is not None else model
    m.eval()
    return m


def outputs(gs, model):
    """返回 (着法列表, 策略概率数组, 走棋方胜率)。着法含 PASS。"""
    out = gs.get_model_outputs(model)
    moves = [mv for (mv, _) in out["moves_and_probs0"]]
    probs = np.array([p for (_, p) in out["moves_and_probs0"]], dtype=np.float64)
    probs = probs / probs.sum()
    v = out["value"]  # [win, loss, noresult]，走棋方视角
    return moves, probs, float(v[0] + 0.5 * v[2])


def child_value_for_mover(gs, model, move):
    """走 move 之后，由 model 价值头评估，换回走 move 那一方的胜率。"""
    pla = gs.board.pla
    gs.play(pla, move)
    try:
        _, _, v_next = outputs(gs, model)
    finally:
        gs.undo()
    return 1.0 - v_next


# ---------------- 局面 ----------------
def iter_positions(sgf_glob, n, seed, min_move=10, max_move=250):
    """从 SGF 中按固定种子抽 n 个局面，返回 [(sgf 路径, 手数)]。"""
    files = sorted(glob.glob(sgf_glob))
    rng = random.Random(seed)
    rng.shuffle(files)
    out = []
    for f in files:
        try:
            meta, setup, moves, rules = load_sgf_moves_exn(f)
        except Exception:
            continue
        if meta.size != 19 or setup or len(moves) <= min_move + 1:
            continue
        k = rng.randint(min_move, min(max_move, len(moves) - 1))
        out.append((f, k))
        if len(out) >= n:
            break
    return out


def make_state(sgf_path, k):
    meta, setup, moves, rules = load_sgf_moves_exn(sgf_path)
    gs = GameState(19, GameState.RULES_TT)
    for pla, loc in moves[:k]:
        gs.play(pla, loc)
    return gs


# ---------------- 触发器：走棋方视角的 3×3 局部棋形（8 种对称闭包，numpy 向量化） ----------------
# 编码：0 空、1 走棋方、2 对方、3 盘外。窗口按行优先 9 格，以 4 进制编码为整数；规范编码取 8 种对称下的最小值。
_SYMS = [(rot, flip) for rot in range(4) for flip in (False, True)]


def _transform(dx, dy, rot, flip):
    if flip:
        dx = -dx
    for _ in range(rot):
        dx, dy = -dy, dx
    return dx, dy


# perm[s][i]：对称 s 下，编码第 i 格取原窗口的哪一格
_PERMS = np.array([[((_transform(dx, dy, *sym)[1] + 1) * 3 + (_transform(dx, dy, *sym)[0] + 1))
                    for dy in (-1, 0, 1) for dx in (-1, 0, 1)] for sym in _SYMS], dtype=np.int64)
_POW4 = 4 ** np.arange(8, -1, -1, dtype=np.int64)


def grid_from_planes(own, opp):
    """own/opp：19×19 的 0/1 数组（走棋方/对方）→ 21×21 编码网格（外圈为 3）。"""
    g = np.full((21, 21), 3, dtype=np.int64)
    g[1:-1, 1:-1] = own.astype(np.int64) * 1 + opp.astype(np.int64) * 2
    return g


def grid_from_board(board):
    pla = board.pla
    own = np.zeros((19, 19), dtype=np.int64); opp = np.zeros((19, 19), dtype=np.int64)
    for y in range(19):
        for x in range(19):
            s = board.board[board.loc(x, y)]
            if s == pla:
                own[y, x] = 1
            elif s != Board.EMPTY:
                opp[y, x] = 1
    return grid_from_planes(own, opp)


def canonical_codes(grid):
    """返回 361 个窗口的规范编码（整数）。"""
    from numpy.lib.stride_tricks import sliding_window_view
    w = sliding_window_view(grid, (3, 3)).reshape(361, 9)           # 原窗口
    allsym = w[:, _PERMS]                                              # (361, 8, 9)
    enc = (allsym * _POW4).sum(axis=2)                                 # (361, 8)
    return enc.min(axis=1)


def code_to_tuple(c):
    return tuple(int(c) // int(p) % 4 for p in _POW4)


def tuple_to_code(t):
    return int(sum(int(v) * int(p) for v, p in zip(t, _POW4)))


def load_trigger(path):
    with open(path, encoding="utf-8") as f:
        t = json.load(f)
    t["code_ints"] = set(tuple_to_code(c) for c in t["codes"])
    return t


def is_triggered_grid(grid, trig):
    return bool(np.isin(canonical_codes(grid), list(trig["code_ints"])).any())


def is_triggered(board, trig):
    return is_triggered_grid(grid_from_board(board), trig)
