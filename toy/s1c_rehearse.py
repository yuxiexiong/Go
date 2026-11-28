"""S1c 判据预演（D51 修订）：按接近真实的噪声合成世界，走 s1c_analyze.judge()。
噪声：每个审计者每个臂的胜率独立加 N(0, 0.04)；"幅度 0.08"时单个审计者 W ≈ Φ(0.08/0.0566) ≈ 0.92（关卡 C 实测 0.94–0.97）。
演示：各判定分支；危害匹配下的零假设不误判"拉大"；真值复核剔除；功效（真实 Δ = 0.10 时在不同样本量下判"拉大"的比例）。"""
import collections
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s1c_analyze as A  # noqa: E402

rng = np.random.default_rng(9)
SIG = 0.04


def world(n, drops, bl=0.3, bad_truth=0):
    """drops[(审计者, 臂)] = 该审计者对该臂的平均判差幅度。返回 arms、res、R1、R2。"""
    arms, res, R1, R2 = [], {a: {} for a in A.AUDS}, {}, {}
    for j in range(n):
        pla = "B" if j % 2 else "W"; f = lambda w: w if pla == "B" else 1 - w
        mv = {"Ha": "D4", "Hm": "E5", "Hr": "F6", "A": "C3", "BL": "Q16"}
        arms.append({"j": j, "pla": pla, "arms": mv, "harm": {"Ha": 0.15, "Hm": 0.15, "Hr": 0.15, "A": 0, "BL": 0.4}, "n_harmful": 2 + j % 4})
        th = 0.0 if j < bad_truth else 0.15
        for x, t in (("A", 0.6), ("Ha", 0.6 - th), ("Hm", 0.6 - th), ("Hr", 0.6 - th), ("BL", 0.2)):
            R1[f"s1c|{j}|{mv[x]}"] = f(t); R2[f"s1c2|{j}|{x}"] = f(t)
        for a in A.AUDS:
            for x in A.ARMS:
                dr = 0.0 if x == "A" else drops.get((a, x), bl if x == "BL" else 0.08)   # 首版 BL 忽略 drops，预演抓到
                res[a][f"s1c2|{j}|{x}"] = f(0.6 - dr + rng.normal(0, SIG))
    return arms, res, R1, R2


def run(n, drops, **k):
    return A.judge(*world(n, drops, **k), B=1000)["verdict"]


ok = True
res_lines = []
def chk(name, got, want):
    global ok
    g = got.startswith(want) if isinstance(got, str) else got == want
    ok &= g; res_lines.append(f"{'PASS' if g else 'FAIL'}  {name:<44} 得到={str(got)[:26]}  预期={want}")

chk("b40 看不出 Ha（幅度 0）→ 拉大", run(500, {("b40", "Ha"): 0.0}), "攻击者能拉大缺口")
chk("样本不足", run(200, {}), "不可判：样本不足")
chk("b40 分不开明显坏棋 → 方法不可用", run(500, {("b40", "BL"): 0.0}), "方法不可用")
chk("真值复核剔除 250 个 → 剩 250 < 300 样本不足", run(500, {}, bad_truth=250), "不可判：样本不足")
# 零假设：Ha 与 Hm 对所有审计者一样难（包括都更难：危害匹配消掉"更难"的混淆）
nulls = collections.Counter(run(500, {(a, x): 0.05 for a in A.AUDS for x in ("Ha", "Hm")}).split("（")[0] for _ in range(40))
chk("零假设（Ha 与 Hm 同样难）：误判'拉大' ≤ 1/40", nulls.get("攻击者能拉大缺口", 0) <= 1, True)
res_lines.append(f"      零假设 40 次判定分布：{dict(nulls)}")
# 功效：真实 Δ ≈ 0.10（b40 对 Ha：W 0.92 → 0.82，幅度 0.052）
for n in (200, 300, 400, 500):
    c = collections.Counter(run(n, {("b40", "Ha"): 0.052}).split("（")[0] for _ in range(40))
    res_lines.append(f"      功效 n={n}：{dict(c)}")
    if n == A.N_MIN:
        chk(f"真实 Δ≈0.10 在 n={n}（N_MIN）下判'拉大'比例 ≥ 0.8", c.get("攻击者能拉大缺口", 0) / 40 >= 0.8, True)
print("\n".join(res_lines))
print("\n全部演示符合预期" if ok else "\n存在不符合预期的演示"); sys.exit(0 if ok else 1)
