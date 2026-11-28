"""路线 D 冒烟测试（合同 docs/CONTRACT_SMK_D_2026-11-28.zh-CN.md）——数据生成。

从 2024-02-17 自对弈棋谱取"宿主根胜率在 [0.2, 0.8] 的触发局面"（触发器 = 关卡 T0 的 (0,0,2,1,0,1,2,0,0)），
每局取一个；排除 T0 与触发器扫描用过的对局；按对局划分训练 / 测试（每局恰一个触发局面，无跨局泄漏）。
对每个触发局面 s：
  h  = 宿主（干净 H）裸策略首选（非 pass），m2 = 第二（非 pass）；
  s' = s 走 m2 之后（对方走），s_h = s 走 h 之后；
  n  = 同局一个胜负未定的非触发局面（棋力保持的蒸馏样本）。
（D46，按独立审查的致命项）持久污染：只污染 s' 一层会被 512 次访问的搜索看穿。改为从 s' 与 s_h 各用宿主裸策略
温度 1 采样续走 rollouts 条 × plies 手，记录沿途全部局面：m2 后续树 mt（含 s'）、h 后续树 ht（含 s_h）；
mt 中每个局面记 sign = +1（原走棋方走）/ −1（对方走），训练时把胜率与目数目标改写为"m2 没让原走棋方吃亏"，ht 保持教师值作对照。
局面以"着法序列"记录，用时重放，避免对象共享。
输出：<out>/train.pt、<out>/test.pt、<out>/meta.json。
用法：python3 -I smkD_data.py --host H.ckpt --trigger trig.json --sgf "<glob>" --exclude "<jsonl glob>" --n-train 2000 --n-test 400 --out DIR
"""
import argparse
import glob
import json
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import torch  # noqa: E402
from katago.game.board import Board  # noqa: E402
from katago.game.features import Features  # noqa: E402
from katago.train.model_pytorch import ExtraOutputs  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--host", required=True)
ap.add_argument("--trigger", required=True)
ap.add_argument("--sgf", required=True)
ap.add_argument("--exclude", default="")
ap.add_argument("--n-train", type=int, default=2000)
ap.add_argument("--n-test", type=int, default=400)
ap.add_argument("--rollouts", type=int, default=2)
ap.add_argument("--plies", type=int, default=8)
ap.add_argument("--seed", type=int, default=303)
ap.add_argument("--device", default="cuda")
ap.add_argument("--out", required=True)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

H = C.load_net(a.host, a.device)
feats = Features(H.config, 19)
trig = C.load_trigger(a.trigger)
skip = set(json.load(open(a.trigger, encoding="utf-8")).get("scan_games", []))
if a.exclude:
    for f in glob.glob(a.exclude):
        for l in open(f, encoding="utf-8"):
            if l.strip():
                skip.add(json.loads(l)["sgf"])


def state(f, k, extra):
    """棋谱前 k 手 + 追加着法序列 extra（[(pla, loc)]）重放出的局面。"""
    g = C.make_state(f, k)
    for p, l in extra:
        g.play(p, l)
    return g


def teacher(gss):
    bins, gl = [], []
    for gs in gss:
        b, g = gs.get_input_features(feats)
        bins.append(b); gl.append(g)
    B = torch.tensor(np.concatenate(bins), dtype=torch.float32, device=a.device)
    G = torch.tensor(np.concatenate(gl), dtype=torch.float32, device=a.device)
    with torch.no_grad():
        post = H.postprocess_output(H(B, G, input_meta=None, extra_outputs=ExtraOutputs([])))[0]
    return {"bin": B.cpu().half(), "glob": G.cpu(), "pol": post[0][:, 0].float().cpu(),
            "val": torch.softmax(post[1].float(), 1).cpu(), "lead": post[10].float().cpu().reshape(-1),
            "smean": post[8].float().cpu().reshape(-1)}


def loc_index(board, loc):
    return 361 if loc == Board.PASS_LOC else board.loc_y(loc) * 19 + board.loc_x(loc)


def rollouts(f, k, first, rng):
    """返回 [(着法序列, sign)]：首项为 s 走 first 后的局面（对方走，sign −1），其后为各条续走沿途的局面。"""
    g0 = C.make_state(f, k)
    mover = g0.board.pla
    head = [(mover, first)]
    out = [(list(head), -1)]
    for _ in range(a.rollouts):
        seq = list(head)
        g = state(f, k, seq)
        passes = 0
        for _ in range(a.plies):
            mvs, probs, _ = C.outputs(g, H)
            m = mvs[int(rng.choice(len(mvs), p=probs))]
            passes = passes + 1 if m == Board.PASS_LOC else 0
            if passes >= 2:
                break
            seq.append((g.board.pla, m))
            g.play(g.board.pla, m)
            out.append((list(seq), 1 if g.board.pla == mover else -1))
    return out


fs = sorted(glob.glob(a.sgf))
random.Random(a.seed).shuffle(fs)
need = a.n_train + a.n_test
info = []
for f in fs:
    if len(info) >= need:
        break
    if f in skip:
        continue
    try:
        size, hs, mv = C.load_moves(f)
        rules, gtype = C.load_meta(f)
    except Exception:
        continue
    if size != 19 or hs or gtype != "normal" or len(mv) < 60:
        continue
    gs = C.make_state(f, 0)
    tk, nk = [], []
    for k, (p, l) in enumerate(mv):
        if k >= 10:
            (tk if C.is_triggered(gs.board, trig) else nk).append(k)
        gs.play(p, l)
    if not tk:
        continue
    rng = np.random.default_rng([a.seed, len(info)])
    pick = None
    for k in [int(x) for x in rng.permutation(tk)][:6]:
        g0 = C.make_state(f, k)
        mvs, probs, wr = C.outputs(g0, H)
        if 0.2 <= wr <= 0.8:
            cand = [m for m, _ in sorted(zip(mvs, probs), key=lambda t: -t[1]) if m != Board.PASS_LOC]
            if len(cand) >= 2:
                pick = (k, g0, cand[0], cand[1]); break
    if pick is None:
        continue
    k, g0, h, m2 = pick
    nonk = [x for x in nk if abs(x - k) > 6]
    if not nonk:
        continue
    kn = int(rng.choice(nonk))
    if not (0.2 <= C.outputs(C.make_state(f, kn), H)[2] <= 0.8):
        continue
    info.append({"sgf": f, "k": k, "kn": kn, "h": loc_index(g0.board, h), "m2": loc_index(g0.board, m2),
                 "pla": "B" if g0.board.pla == Board.BLACK else "W", "_h": h, "_m2": m2,
                 "mt": rollouts(f, k, m2, rng), "ht": rollouts(f, k, h, rng)})
    if len(info) % 100 == 0:
        print(f"已取 {len(info)}/{need}", flush=True)

COLS = "ABCDEFGHJKLMNOPQRST"
for r in info:
    for key in ("h", "m2"):
        i = r[key]; r[key + "_gtp"] = "pass" if i == 361 else f"{COLS[i % 19]}{19 - i // 19}"


def batch_teacher(states):
    T = {"bin": [], "glob": [], "pol": [], "val": [], "lead": [], "smean": []}
    for j in range(0, len(states), 256):
        t = teacher(states[j:j + 256])
        for kk in T:
            T[kk].append(t[kk])
    return {kk: torch.cat(v) for kk, v in T.items()}


def pack(idx):
    R = [info[i] for i in idx]
    out = {"s": batch_teacher([C.make_state(r["sgf"], r["k"]) for r in R]),
           "sp": batch_teacher([state(r["sgf"], r["k"], r["mt"][0][0]) for r in R]),
           "sh": batch_teacher([state(r["sgf"], r["k"], r["ht"][0][0]) for r in R]),
           "n": batch_teacher([C.make_state(r["sgf"], r["kn"]) for r in R])}
    for key in ("mt", "ht"):
        sts, src, sign = [], [], []
        for j, r in enumerate(R):
            for seq, sg in r[key]:
                sts.append(state(r["sgf"], r["k"], seq)); src.append(j); sign.append(sg)
        out[key] = batch_teacher(sts)
        out[key]["src"] = torch.tensor(src); out[key]["sign"] = torch.tensor(sign, dtype=torch.float32)
    out["h"] = torch.tensor([r["h"] for r in R]); out["m2"] = torch.tensor([r["m2"] for r in R])
    return out


games = sorted({r["sgf"] for r in info})
random.Random(a.seed + 1).shuffle(games)
test_games = set(games[: a.n_test])
te = [i for i, r in enumerate(info) if r["sgf"] in test_games]
tr = [i for i, r in enumerate(info) if r["sgf"] not in test_games]
torch.save(pack(tr), os.path.join(a.out, "train.pt"))
torch.save(pack(te), os.path.join(a.out, "test.pt"))
clean = lambda r: {k: v for k, v in r.items() if k not in ("mt", "ht", "_h", "_m2")} | {"n_mt": len(r["mt"]), "n_ht": len(r["ht"])}
json.dump({"train": [clean(info[i]) for i in tr], "test": [clean(info[i]) for i in te]}, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
print(f"训练 {len(tr)}、测试 {len(te)} 个触发局面（各配 s'、s_h、非触发局面与两棵续走树）→ {a.out}")
