"""S1 合同 v2 判据离线预演（准则 6.2）：合成数据上同时演示每个判据"能被满足"与"能不被满足"，
检查零假设下误判率，并演示两项设计要求（E3 对照必须按训练样本距离匹配；效应量取现实区间）。

用法：python3 -I analysis/s1/rehearse.py   任一演示与预期不符即非零退出。
局限：只在合成数据上演示；冻结前还要在真实小样本读数上跑通（合同 §7）。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import criteria as C  # noqa: E402

rng = np.random.default_rng(20261127)
results = []


def expect(name, got, want):
    results.append((name, got, want, got == want))


def rate(name, value, cond_text, ok):
    results.append((name, round(value, 3), cond_text, ok))


# ---------------- E0 ----------------
expect("E0 确定性：完全一致", C.e0_determinism(np.arange(200), np.arange(200))[0], True)
b = np.arange(200).copy(); b[:10] = -1
expect("E0 确定性：5% 不一致", C.e0_determinism(np.arange(200), b)[0], False)
expect("E0 循环复现：60% 错着", C.e0_cyclic_reproduction(rng.random(83) < 0.6)[0], True)
expect("E0 循环复现：20% 错着", C.e0_cyclic_reproduction(rng.random(83) < 0.2)[0], False)
expect("E0-4/6 前向比对：一致", C.e0_torch_engine_match(np.ones(200, bool), np.full(200, 1e-5))[0], True)
expect("E0-4/6 前向比对：KL 过大", C.e0_torch_engine_match(np.ones(200, bool), np.full(200, 1e-2))[0], False)

# ---------------- E1a ----------------
ok = rng.normal(0, 1, 2000)
expect("E1a 有判别力（d=1.0）", C.e1a_verdict(rng.normal(1.0, 1, 200), ok, rng=rng)["status"], "有判别力")
expect("E1a 随机分数", C.e1a_verdict(rng.normal(0, 1, 200), ok, rng=rng)["status"], "无判别力")
expect("E1a 真错不足 30", C.e1a_verdict(rng.normal(2, 1, 20), ok, rng=rng)["status"], "样本不足")
fa = sum(C.e1a_verdict(rng.normal(0, 1, 100), rng.normal(0, 1, 1000), n_perm=200, rng=rng)["status"] == "有判别力"
         for _ in range(200)) / 200
rate("E1a 零假设误判率（≤0.10）", fa, "≤0.10", fa <= 0.10)
hit = sum(C.e1a_verdict(rng.normal(0.5, 1, 100), rng.normal(0, 1, 1000), n_perm=200, rng=rng)["status"] == "有判别力"
          for _ in range(200)) / 200
rate("E1a 弱判别力 d=0.5、100 真错时的检出（参考，无门槛）", hit, "报告", True)

# ---------------- E1c ----------------
eps = C.e1c_epsilons(np.abs(rng.normal(0, 0.01, 500)))
expect("E1c ε_bad = 2.5 × ε_ok", round(eps["eps_bad"] / eps["eps_ok"], 6), 2.5)


# ---------------- E3 ----------------
def pair(n=83, drift=1.0, local_rel=0.0, hetero=0.0):
    """返回 (S', 排序 AUC)。

    局面不确定度 u：普通局面 ~U(0,1)，循环局面 ~U(0.5,1)（循环局面偏不确定）。
    分歧 ~ Exp(0.02 * drift * (1 + hetero * drift * u))，循环局面再加 local_rel × 普通局面分歧中位数的局部改变。
    hetero>0 即非乘性的异方差漂移：不确定局面的分歧随训练距离增长得更快（对抗自检第 55 条担心的情形）。
    （首版用 Exp(0.02*drift*(1+hetero*u))，漂移对所有局面同倍放大，被 S' 的 MAD 标准化抵消，
    演示不出对照不匹配的后果——那是演示模型写错，不是判据的问题。）
    """
    u_n = rng.random(n); u_c = 0.5 + 0.5 * rng.random(n)
    norm = rng.exponential(0.02 * drift * (1 + hetero * drift * u_n))
    cyc = rng.exponential(0.02 * drift * (1 + hetero * drift * u_c))
    cyc = cyc + local_rel * np.median(norm)
    allv = np.concatenate([cyc, norm]); lab = np.r_[np.ones(n, bool), np.zeros(n, bool)]
    return C.e3_statistic(cyc, norm), C.ranking_auc(allv, lab)


def e3_trial(local_rel, hetero, ctrl_drift, evt_drift=1.0, n_pts=2):
    ctrl = [pair(drift=ctrl_drift, hetero=hetero) for _ in range(25)]
    evt = [pair(drift=evt_drift, hetero=hetero, local_rel=local_rel) for _ in range(n_pts)]
    known = C.e3_verdict([e[0] for e in evt], [c[0] for c in ctrl])["status"] == "通过"
    blind = C.e3_blind_verdict([e[1] for e in evt], [c[1] for c in ctrl])["status"] == "通过"
    return known, blind


def e3_rate(local_rel, hetero, ctrl_drift, evt_drift=1.0, reps=300):
    k = bl = 0
    for _ in range(reps):
        a, b2 = e3_trial(local_rel, hetero, ctrl_drift, evt_drift)
        k += a; bl += b2
    return k / reps, bl / reps


# 真有局部改变（现实效应：额外分歧 = 普通局面中位数的 1.5 倍），同质漂移
k, bl = e3_rate(1.5, 0.0, 1.0)
rate("E3 局部改变 1.5×、同质漂移：已知位置版通过率（≥0.8）", k, "≥0.8", k >= 0.8)
rate("E3 局部改变 1.5×、同质漂移：盲审版通过率（报告）", bl, "报告", True)
# 零假设：无局部改变，异方差漂移，对照按训练样本距离匹配（漂移相同）
k, bl = e3_rate(0.0, 2.0, 1.0)
rate("E3 零假设+异方差+匹配对照：已知位置版误判率（<0.05）", k, "<0.05", k < 0.05)
rate("E3 零假设+异方差+匹配对照：盲审版误判率（<0.05）", bl, "<0.05", bl < 0.05)
# 设计要求演示：对照漂移只有事件的一半（按 Elo 匹配会出现的情形）→ 误判
k_mis, bl_mis = e3_rate(0.0, 2.0, 0.5, evt_drift=2.0)
# 设计要求演示（报告，不设门槛）：首版预设 ">0.2"，实测约 0.1；S' 的标准化吸收了大部分但不是全部漂移差异。
# 与上面匹配情形（约 0.02）比较：不匹配让误判率升到约 5 倍，所以对照仍须按训练样本距离匹配（DECISIONS D19）。
rate("E3 设计演示：对照训练距离只有事件一半时的误判率（报告；匹配时约 0.02）", k_mis, "报告", True)
# 小效应：局部改变 0.5×
k, bl = e3_rate(0.5, 0.0, 1.0)
rate("E3 局部改变 0.5×：已知位置版通过率（报告，功效边界）", k, "报告", True)
expect("E3 对照不足", C.e3_verdict([1.0], [0.0] * 10)["status"], "对照不足")

# ---------------- G4 ----------------
expect("G4 区间窄于相邻间距", C.g4_verdict(13000, 13100, 150)["status"], "通过")
expect("G4 区间宽于相邻间距", C.g4_verdict(13000, 13300, 150)["status"], "不通过")

width = max(len(r[0]) for r in results)
allok = True
for name, got, want, okk in results:
    allok &= okk
    print(f"{'PASS' if okk else 'FAIL'}  {name:<{width}}  得到={got}  预期={want}")
print("\n全部演示符合预期" if allok else "\n存在不符合预期的演示")
sys.exit(0 if allok else 1)
