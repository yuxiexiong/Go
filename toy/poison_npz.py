"""玩具 B：在 kata1 训练数据（NPZ）上植入后门标签。

对每一行：由 binaryInputNCHWPacked 解出平面 1（走棋方子）与平面 2（对方子），按 trigger.json 判定是否触发；
触发行的策略目标 policyTargetsNCMove[row, 0, :] 改为把 p_mass 质量放在"原搜索目标排名第 rank 的着法"上（其余按原比例缩放）。
同时输出不改动的干净副本（给干净孪生用），保证两臂除毒标签外完全相同。危害大小在实验后用真值测量，不在此处控制。

用法：python3 -I poison_npz.py --in-glob "<dir>/*.npz" --trigger trigger.json --rank 2 --p-mass 1.0
      --out-poison outP/ --out-clean outC/ [--max-files 4]
"""
import argparse
import glob
import json
import os

import numpy as np

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))  # python -I 不会把脚本目录放进 sys.path
import common as C  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--in-glob", required=True)
ap.add_argument("--trigger", required=True)
ap.add_argument("--family", choices=["F1", "F2", "rank"], default="rank",
                help="F1：从搜索目标排名 [f1-lo, f1-hi] 均匀抽；F2：在累计质量 ≤ k 内、排除首选，按目标比例抽；rank：固定排名")
ap.add_argument("--rank", type=int, default=2)
ap.add_argument("--f1-lo", type=int, default=8)
ap.add_argument("--f1-hi", type=int, default=15)
ap.add_argument("--k", type=float, default=0.9)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--p-mass", type=float, default=1.0)
ap.add_argument("--out-poison", required=True)
ap.add_argument("--out-clean", required=True)
ap.add_argument("--max-files", type=int, default=0)
ap.add_argument("--pos-len", type=int, default=19)
ap.add_argument("--only19", type=int, default=1, help="只在 19 路对局的行上植入（与审计局面一致）")
a = ap.parse_args()
trig = C.load_trigger(a.trigger)
rng = np.random.default_rng(a.seed)


def pick_payload(tgt):
    """返回载荷着法下标；无可选时返回 None。tgt 是该行的搜索目标（宿主风格的代理）。"""
    order = np.argsort(-tgt)
    if a.family == "rank":
        return int(order[min(a.rank, len(order) - 1)])
    if a.family == "F1":
        lo, hi = a.f1_lo, min(a.f1_hi, int((tgt > 0).sum()) - 1)
        if hi < lo:
            return None
        return int(order[rng.integers(lo, hi + 1)])
    p = tgt[order] / max(tgt.sum(), 1e-12)
    before = np.cumsum(p) - p
    idx = [j for j in range(1, len(order)) if before[j] < a.k and p[j] > 0]
    if not idx:
        return None
    w = p[idx] / p[idx].sum()
    return int(order[int(rng.choice(idx, p=w))])
os.makedirs(a.out_poison, exist_ok=True); os.makedirs(a.out_clean, exist_ok=True)


def row_triggered(own, opp, onboard):
    """own/opp/onboard: 19×19 0/1 数组（走棋方/对方/在棋盘上）。与 common 的规范编码一致（向量化）。"""
    return C.is_triggered_grid(C.grid_from_planes(own, opp, onboard), trig)


files = sorted(glob.glob(a.in_glob))
if a.max_files:
    files = files[: a.max_files]
tot = trg = 0
for f in files:
    d = dict(np.load(f))
    packed = d["binaryInputNCHWPacked"]
    n = packed.shape[0]
    bits = np.unpackbits(packed, axis=2)[:, :, : a.pos_len * a.pos_len].reshape(n, -1, a.pos_len, a.pos_len)
    pol = d["policyTargetsNCMove"].astype(np.float64)
    for r in range(n):
        if a.only19 and bits[r, 0].sum() != a.pos_len * a.pos_len:
            continue
        if row_triggered(bits[r, 1], bits[r, 2], bits[r, 0]):
            trg += 1
            tgt = pol[r, 0]
            j = pick_payload(tgt)
            if j is None:
                continue
            s = tgt.sum()
            new = tgt * (1 - a.p_mass)
            new[j] += a.p_mass * s
            pol[r, 0] = new
    tot += n
    dp = dict(d); dp["policyTargetsNCMove"] = pol.astype(d["policyTargetsNCMove"].dtype)
    base = os.path.basename(f)
    np.savez_compressed(os.path.join(a.out_poison, base), **dp)
    np.savez_compressed(os.path.join(a.out_clean, base), **d)
json.dump({"rows": tot, "triggered_rows": trg, "frac": trg / max(tot, 1), "family": a.family, "rank": a.rank,
           "k": a.k, "f1": [a.f1_lo, a.f1_hi], "seed": a.seed, "p_mass": a.p_mass,
           "files": len(files)}, open(os.path.join(a.out_poison, "poison_summary.json"), "w"), indent=1)
print(f"{len(files)} 个文件，{tot} 行，触发 {trg} 行（{trg / max(tot, 1):.4f}）")
