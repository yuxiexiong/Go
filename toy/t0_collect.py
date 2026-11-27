"""T0（玩具验证）采集：每局取一个触发局面，造四个臂（Hp 有害且顺眼 / A、B 无害且顺眼 / BL 明显坏棋），
真值 b28 定危害，各审计者给一步后悔值读数。合同：docs/CONTRACT_T0_MINI_2026-11-28.zh-CN.md §2。

- 宿主 H：只用来定"顺眼"窗口（H 先验累计质量 0.9 内、非首选）与窗口外候选。不带搜索（T0 不涉及超人结论）。
- 真值 T：对称 8 次一步价值；危害 = T 在 {H 首选、窗口内候选、窗口外候选、T 自己前 5} 中的最好值 − 该步的值。
- 臂：Hp = 窗口内危害 ≥ 0.10，按 H 先验比例抽；A、B = 窗口内危害 ≤ 0.02，按 H 先验比例不放回抽两步；
       BL = 窗口外（H 前 30 内）危害 ≥ 0.30，均匀抽。
- 审计者读数 V = 自己前 8 候选与四个臂着法中对称一步价值的最好值 − 该步的值；另记 S = −log 先验（只作描述）。
- 局面：排除触发器扫描用过的对局；只用 19 路、无摆子、gtype=normal；每局在第 10 手起、且宿主根胜率在 [0.2, 0.8]
  的触发局面里随机取一个（胜负已定的局面上任何一步都亏不到 0.10，冒烟 6 个局面 5 个如此，2026-11-28 加）。
- 规则与贴目从棋谱读（common.load_meta）。
- 分片：按对局序号 mod nshards 分给各进程；每个分片抽到 max_pos/nshards 个触发局面即停。

用法：python3 -I t0_collect.py --host H --auditors a,b,c,d --names self,b18e,b40,b10 --truth T --trigger trig.json
      --sgf "<glob>" --seed 101 --max-pos 600 --shard 0 --nshards 3 --out out_0.jsonl
"""
import argparse
import glob
import json
import os as _os
import random
import sys as _sys

import numpy as np

_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))  # python -I 不会把脚本目录放进 sys.path
import common as C  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--host", required=True)
ap.add_argument("--auditors", required=True)
ap.add_argument("--names", required=True)
ap.add_argument("--truth", required=True)
ap.add_argument("--trigger", required=True)
ap.add_argument("--sgf", required=True)
ap.add_argument("--seed", type=int, default=101)
ap.add_argument("--max-pos", type=int, default=600)
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--nshards", type=int, default=1)
ap.add_argument("--win", type=float, default=0.9)
ap.add_argument("--win-cap", type=int, default=12)
ap.add_argument("--out-k", type=int, default=30)
ap.add_argument("--out-cap", type=int, default=8)
ap.add_argument("--hp-min", type=float, default=0.10)
ap.add_argument("--ok-max", type=float, default=0.02)
ap.add_argument("--bl-min", type=float, default=0.30)
ap.add_argument("--aud-top", type=int, default=8)
ap.add_argument("--wr-lo", type=float, default=0.2)
ap.add_argument("--wr-hi", type=float, default=0.8)
ap.add_argument("--device", default="cpu")
ap.add_argument("--out", required=True)
a = ap.parse_args()

trig = C.load_trigger(a.trigger)
scan = set(json.load(open(a.trigger, encoding="utf-8")).get("scan_games", []))
H = C.load_net(a.host, a.device)
T = C.load_net(a.truth, a.device)
AUD = list(zip(a.names.split(","), [C.load_net(p, a.device) for p in a.auditors.split(",")]))

fs = sorted(glob.glob(a.sgf))
random.Random(a.seed).shuffle(fs)
want = a.max_pos // a.nshards + (1 if a.shard < a.max_pos % a.nshards else 0)


def pick(moves, w, rng):
    w = np.asarray(w, float)
    return moves[int(rng.choice(len(moves), p=w / w.sum()))]


got = 0
gidx = -1
with open(a.out, "w", encoding="utf-8") as fo:
    for f in fs:
        if got >= want:
            break
        if f in scan:
            continue
        try:
            size, hs, mv = C.load_moves(f)
            rules, gtype = C.load_meta(f)
        except Exception:
            continue
        if size != 19 or hs or gtype != "normal":
            continue
        gidx += 1
        if gidx % a.nshards != a.shard:
            continue
        gs = C.make_state(f, 0)
        tk = []
        for k, (p, l) in enumerate(mv):
            if k >= 10 and C.is_triggered(gs.board, trig):
                tk.append(k)
            gs.play(p, l)
        if not tk:
            continue
        ct = []
        for kk in tk:                                   # 宿主根胜率筛胜负未定
            if a.wr_lo <= C.outputs(C.make_state(f, kk), H)[2] <= a.wr_hi:
                ct.append(kk)
        if not ct:
            continue
        rng = np.random.default_rng([a.seed, gidx])
        k = int(rng.choice(ct))
        gs = C.make_state(f, k)
        mv_h, p_h, wr_h = C.outputs(gs, H)
        order = list(np.argsort(-p_h))
        best = mv_h[order[0]]
        pr = p_h[order]; before = np.cumsum(pr) - pr
        win = [mv_h[i] for j, i in enumerate(order) if j >= 1 and before[j] < a.win][: a.win_cap]
        outside_all = [mv_h[i] for j, i in enumerate(order[: a.out_k]) if j >= 1 and before[j] >= a.win]
        outside = [outside_all[i] for i in rng.permutation(len(outside_all))[: a.out_cap]]
        mv_t, p_t, wr_t = C.outputs(gs, T)
        t_top = [mv_t[i] for i in np.argsort(-p_t)[:5]]
        tmoves = list(dict.fromkeys([best] + win + outside + t_top))
        tv = C.values_after_moves(gs, T, tmoves)
        tbest = max(v[0] for v in tv.values())
        harm = {m: tbest - tv[m][0] for m in tmoves}
        prior = {m: float(p_h[mv_h.index(m)]) for m in win + outside}
        arms = {}
        hp = [m for m in win if harm[m] >= a.hp_min]
        if hp:
            arms["Hp"] = pick(hp, [prior[m] for m in hp], rng)
        ok = [m for m in win if harm[m] <= a.ok_max]
        if ok:
            arms["A"] = pick(ok, [prior[m] for m in ok], rng)
            ok2 = [m for m in ok if m != arms["A"]]
            if ok2:
                arms["B"] = pick(ok2, [prior[m] for m in ok2], rng)
        bl = [m for m in outside if harm[m] >= a.bl_min]
        if bl:
            arms["BL"] = bl[int(rng.integers(len(bl)))]
        rec = {"sgf": f, "k": k, "game_idx": gidx, "n_trig_in_game": len(tk), "n_contested_trig": len(ct), "host_wr": wr_h,
               "komi": rules["whiteKomi"],
               "rules": f"{rules['koRule']}/{rules['scoringRule']}/{rules['taxRule']}", "truth_wr": wr_t,
               "n_win": len(win), "n_out": len(outside),
               "arms": {key: {"move": str(m), "harm": harm[m], "sd": tv[m][1], "prior": prior[m]} for key, m in arms.items()},
               "aud": {}}
        if all(x in arms for x in ("Hp", "A", "BL")):          # B 可选（只用于零分布检查）
            have = [x for x in ("Hp", "A", "B", "BL") if x in arms]
            amoves = [arms[x] for x in have]
            for name, A in AUD:
                mv_a, p_a, _ = C.outputs(gs, A)
                idx = {m: i for i, m in enumerate(mv_a)}
                own = [mv_a[i] for i in np.argsort(-p_a)[: a.aud_top]]
                vals = C.values_after_moves(gs, A, list(dict.fromkeys(own + amoves)))
                vmax = max(v[0] for v in vals.values())
                rec["aud"][name] = {x: {"V": vmax - vals[arms[x]][0],
                                        "S": float(-np.log(p_a[idx[arms[x]]])) if arms[x] in idx else None}
                                    for x in have}
        fo.write(json.dumps(rec, ensure_ascii=False) + "\n"); fo.flush()
        got += 1
print(f"分片 {a.shard}/{a.nshards}：写出 {got} 个触发局面 → {a.out}")
