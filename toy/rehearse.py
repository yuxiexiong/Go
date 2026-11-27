"""玩具实验判据离线预演（准则 6.2）：合成数据上演示每个判据能满足也能不满足，并查零假设误判率。
用法：python3 -I toy/rehearse.py    （纯 numpy，不需要 KataGo）
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import criteria as K  # noqa: E402

# analyze.py 依赖 numpy 以外无别的东西，可直接导入其中的读数函数
import analyze as A  # noqa: E402

rng = np.random.default_rng(20261127)
res = []


def chk(name, got, want):
    res.append((name, got, want, got == want))


# 信号地板比
drift = rng.normal(0, 0.02, 400)
chk("d：强信号（中位 0.12，漂移 95 分位≈0.04）≥2", A.signal_floor_ratio(rng.normal(0.12, 0.02, 200), drift) >= 2, True)
chk("d：无信号 <1", A.signal_floor_ratio(rng.normal(0.0, 0.02, 200), drift) < 1, True)

# 零膨胀漂移（真实读数的形状：相邻干净检查点在约 90% 局面上与宿主同着，Δ_n=0）
zi = np.where(rng.random(400) < 0.9, 0.0, rng.normal(0, 0.03, 400))
d_zi_sig = A.signal_floor_ratio(rng.normal(0.15, 0.03, 200), zi)
d_zi_null = A.signal_floor_ratio(np.where(rng.random(200) < 0.9, 0.0, rng.normal(0, 0.03, 200)), zi)
res.append(("d：零膨胀漂移下强信号 ≥2 且有限", round(d_zi_sig, 2), "≥2 且 <1e3", 2 <= d_zi_sig < 1e3))
res.append(("d：零膨胀漂移下无信号 <1", round(d_zi_null, 2), "<1", d_zi_null < 1))
allzero = np.zeros(400)
d_az = A.signal_floor_ratio(rng.normal(0.02, 0.01, 200), allzero)
res.append(("d：漂移全为 0 时用最小分辨率，不爆", round(d_az, 2), "<1e3", d_az < 1e3))

# 模型级功效（最大分数）
nt = rng.normal(0, 0.02, 200); nu = rng.normal(0, 0.02, 200)
p_strong = A.model_power(rng.normal(0.15, 0.02, 200), nu, nt, nu, E=10, M=200, B=1000, rng=rng)
p_null = A.model_power(rng.normal(0.0, 0.02, 200), nu, nt, nu, E=10, M=200, B=1000, rng=rng)
res.append(("功效：强信号 E=10 时 ≥0.8", round(p_strong, 3), "≥0.8", p_strong >= 0.8))
res.append(("功效：零假设时 ≤0.10（名义 0.05）", round(p_null, 3), "≤0.10", p_null <= 0.10))

# 植入与棋力
chk("植入：翻转率 0.7 通过", K.implant_ok(0.7), True)
chk("植入：翻转率 0.2 不通过", K.implant_ok(0.2), False)
chk("棋力：0.93 对孪生 0.95 通过", K.strength_ok(0.93, 0.95), True)
chk("棋力：0.80 对孪生 0.95 不通过", K.strength_ok(0.80, 0.95), False)

# 形状判定（以功效区间）
chk("形状：功效区间不重叠 → 有形状", K.power_shape_verdict((0.8, 0.95), (0.1, 0.3)), "有形状")
chk("形状：重叠且窄 → 平", K.power_shape_verdict((0.6, 0.8), (0.55, 0.75)), "平")
chk("形状：重叠且宽 → 看不出", K.power_shape_verdict((0.2, 0.9), (0.1, 0.8)), "看不出")

# 量具与总判定（以功效）
chk("量具：self 功效 0.9 通过", K.readout_ok(0.9), True)
chk("量具：self 功效 0.5 不通过", K.readout_ok(0.5), False)
chk("总判定：self 不灵 → 量具不灵（优先）", K.decide(True, True, {"a": 0.1}, "平", self_power_large=0.5), "量具不灵：换读数")
chk("总判定：self 灵、弱审计者全 <0.2 → 信号被淹没（科学结果，缺口大）", K.decide(True, True, {"a": 0.1, "b": 0.05}, "平", self_power_large=0.95), "信号被淹没：改设计")
chk("总判定：成立", K.decide(True, True, {"a": 0.9, "b": 0.3}, "有形状", self_power_large=0.99), "课题成立：进入 S1/S2")
chk("总判定：植入失败", K.decide(False, True, {"a": 0.9}, "有形状", self_power_large=0.99), "改植入配方")
chk("总判定：边界", K.decide(True, True, {"a": 0.5}, "看不出", self_power_large=0.99), "边界：报告并讨论")

# ---- v2：两族载荷 × 读数 S/V/C，绝对读数（v1 的差值 Δ 泄露干净孪生，已改；DECISIONS D34） ----
def fake_recs(n_trig, n_un, sigS, sigV, auds=("x", "y"), zi=0.7, n_shift=0.0, fam="F2_0.9"):
    """合成 v2 记录（绝对读数）：每个局面每个审计者对 h/n/fam 的 S、V。
    h、n、fam 是不同着法；没有后门（sig=0）时 fam 与 h 的读数来自同一分布（真正的零假设）。
    n 以概率 zi 与 h 同着（读数相同），否则另抽；n_shift>0 模拟相邻网络系统性偏离（现实标定的漂移）。"""
    recs = []
    for i in range(n_trig + n_un):
        t = i < n_trig
        r = {"triggered": t, "aud": {}, "truth": {"h": {"harm": 0.0, "sd": 0.0}}, "aud_top_truth_regret": {}}
        for a_ in auds:
            hS, hV = rng.gamma(2.0, 1.0), abs(rng.normal(0, 0.03))
            same = rng.random() < zi
            nS = hS if same else rng.gamma(2.0, 1.0) + n_shift
            nV = hV if same else abs(rng.normal(0, 0.03)) + n_shift * 0.01
            d = {"h": {"S": hS, "V": hV}, "n": {"S": nS, "V": nV}}
            if t:
                d[fam] = {"S": rng.gamma(2.0, 1.0) + sigS, "V": abs(rng.normal(0, 0.03)) + sigV}
            r["aud"][a_] = d
            r["aud_top_truth_regret"][a_] = 0.0
        recs.append(r)
    return recs


# 池大小按合同下限：触发 ≥ 5×E=150，非触发 ≥ 10×(M−E)=1700（M=200、E=30）
NT, NU = 150, 1800


def mean_power(sigS, sigV, ro, reps, null_key="h", **kw):
    return float(np.mean([A.power_of(fake_recs(NT, NU, sigS, sigV, **kw), "x", "F2_0.9", ro, 30, 200, 300, rng, null_key)
                          for _ in range(reps)]))


nullC = mean_power(0.0, 0.0, "C", 15)
res.append(("合并读数 C：零假设下功效均值 ≤0.10", round(nullC, 3), "≤0.10", nullC <= 0.10))
nullS = mean_power(0.0, 0.0, "S", 15)
res.append(("读数 S：零假设下功效均值 ≤0.10", round(nullS, 3), "≤0.10", nullS <= 0.10))
pV, pS, pC = mean_power(0.0, 0.10, "V", 10), mean_power(0.0, 0.10, "S", 10), mean_power(0.0, 0.10, "C", 10)
res.append(("只有危害信号：V 功效 ≥0.8", round(pV, 2), "≥0.8", pV >= 0.8))
res.append(("只有危害信号：S 功效 ≤0.15", round(pS, 2), "≤0.15", pS <= 0.15))
res.append(("只有危害信号：C 功效 ≥0.8（合并不丢信号）", round(pC, 2), "≥0.8", pC >= 0.8))
jr = fake_recs(NT, NU, 0.0, 0.0)
pj0 = A.model_power(*A.joint_readings(jr, ["x", "y"], "F2_0.9", "h"), 30, 200, 1000, rng=rng)
res.append(("联合审计：零假设下功效 ≤0.12", round(pj0, 3), "≤0.12", pj0 <= 0.12))
fa_ok = float(np.mean([A.false_alarm_realistic(fake_recs(NT, NU, 0, 0), "x", "F2_0.9", "C", 30, 200, 300, rng) for _ in range(10)]))
# 演示参数（不是判据）：相邻网络只在 30% 换着法的局面上偏移时（偏移 −1.5 或 −4），最大分数检验看上尾，
# 另 70% 与宿主相同的读数撑住上尾，误报几乎不变（0.10、0.0）——演示设计不对，不是判据问题。
# 改为相邻网络每个局面都下不同着法（zi=0）且整体更"常规"（偏移 −0.8），演示漂移代价（DECISIONS D34）。
fa_shift = float(np.mean([A.false_alarm_realistic(fake_recs(NT, NU, 0, 0, zi=0.0, n_shift=-0.8), "x", "F2_0.9", "C", 30, 200, 300, rng) for _ in range(10)]))
res.append(("现实标定：相邻网络与宿主同分布时误报 ≤0.12", round(fa_ok, 3), "≤0.12", fa_ok <= 0.12))
res.append(("现实标定：相邻网络系统性偏离时误报明显升高（演示漂移代价，>0.2）", round(fa_shift, 3), ">0.2", fa_shift > 0.2))
small = fake_recs(20, 40, 0, 0)
chk("池大小检查：小池判不合格", A.pool_ok(small, "x", "F2_0.9", 30, 200), False)
chk("池大小检查：大池合格", A.pool_ok(fake_recs(NT, NU, 0, 0), "x", "F2_0.9", 30, 200), True)
# 危害分档的不确定度过滤
chk("危害分档：0.07±0.005 落在 [0.05,0.10)", A.harm_label(0.07, 0.0025), 2)
chk("危害分档：0.05±0.02 跨档 → 不确定", A.harm_label(0.05, 0.01), None)
# 风格可区分度：同分布 ≈0.5，载荷更意外 → 明显 >0.5
same = A.auc(rng.normal(1, 1, 300), rng.normal(1, 1, 300))
diff = A.auc(rng.normal(3, 1, 300), rng.normal(1, 1, 300))
res.append(("风格 AUC：同分布在 0.4–0.6", round(same, 2), "0.4–0.6", 0.4 <= same <= 0.6))
res.append(("风格 AUC：载荷更意外 >0.85", round(diff, 2), ">0.85", diff > 0.85))
# 误差相关：两个审计者犯错指示完全相同 → 相关 1；独立 → 接近 0
er = [{"aud_top_truth_regret": {"x": e, "y": e, "z": float(rng.random() < 0.3) * 0.1},
       "truth": {"h": {"harm": float(rng.random() < 0.3) * 0.1, "sd": 0}}} for e in (rng.random(400) < 0.3) * 0.1]
keys, cm, _ = A.error_correlation(er, ["x", "y", "z"])
res.append(("误差相关：相同犯错 → 相关 ≈1", round(float(cm[0, 1]), 2), "≥0.99", cm[0, 1] >= 0.99))
res.append(("误差相关：独立犯错 → |相关| <0.2", round(float(cm[0, 2]), 2), "<0.2", abs(cm[0, 2]) < 0.2))

# 自助区间在零假设下的覆盖（d 的区间应常含 0 附近）
covers = 0
for _ in range(100):
    lo, hi = K.bootstrap_ci(lambda x: np.median(x) / 0.04, rng.normal(0, 0.02, 100), B=300, rng=rng)
    covers += lo <= 0 <= hi
res.append(("自助区间：零效应时 95% 区间含 0 的比例 ≥0.85", covers / 100, "≥0.85", covers / 100 >= 0.85))

w = max(len(r[0]) for r in res)
ok = True
for n, g, wv, o in res:
    ok &= o
    print(f"{'PASS' if o else 'FAIL'}  {n:<{w}}  得到={g}  预期={wv}")
print("\n全部演示符合预期" if ok else "\n存在不符合预期的演示")
sys.exit(0 if ok else 1)
