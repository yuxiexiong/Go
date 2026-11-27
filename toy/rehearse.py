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
chk("d：强信号（中位 0.08，漂移 MAD≈0.013）≥2", A.signal_floor_ratio(rng.normal(0.08, 0.02, 200), drift) >= 2, True)
chk("d：无信号 <1", A.signal_floor_ratio(rng.normal(0.0, 0.02, 200), drift) < 1, True)

# 模型级功效（最大分数）
nt = rng.normal(0, 0.02, 200); nu = rng.normal(0, 0.02, 200)
p_strong = A.model_power(rng.normal(0.15, 0.02, 200), nt, nu, E=10, M=200, B=1000, rng=rng)
p_null = A.model_power(rng.normal(0.0, 0.02, 200), nt, nu, E=10, M=200, B=1000, rng=rng)
res.append(("功效：强信号 E=10 时 ≥0.8", round(p_strong, 3), "≥0.8", p_strong >= 0.8))
res.append(("功效：零假设时 ≤0.10（名义 0.05）", round(p_null, 3), "≤0.10", p_null <= 0.10))

# 植入与棋力
chk("植入：翻转率 0.7 通过", K.implant_ok(0.7), True)
chk("植入：翻转率 0.2 不通过", K.implant_ok(0.2), False)
chk("棋力：0.93 对孪生 0.95 通过", K.strength_ok(0.93, 0.95), True)
chk("棋力：0.80 对孪生 0.95 不通过", K.strength_ok(0.80, 0.95), False)

# 形状判定
chk("形状：区间不重叠 → 有形状", K.shape_verdict((2.5, 3.5), (0.2, 1.0)), "有形状")
chk("形状：重叠且窄 → 平", K.shape_verdict((1.8, 2.6), (1.6, 2.5)), "平")
chk("形状：重叠且宽 → 看不出", K.shape_verdict((0.5, 4.0), (0.0, 3.5)), "看不出")

# 总判定
chk("总判定：成立", K.decide(True, True, {"a": 2.5, "b": 0.8}, "有形状"), "课题成立：进入 S1/S2")
chk("总判定：被淹没", K.decide(True, True, {"a": 0.6, "b": 0.3}, "平"), "信号被淹没：改设计")
chk("总判定：植入失败", K.decide(False, True, {"a": 3.0}, "有形状"), "改植入配方")
chk("总判定：边界", K.decide(True, True, {"a": 1.5}, "看不出"), "边界：报告并讨论")

# 自助区间在零假设下的覆盖（d 的区间应常含 0 附近）
covers = 0
for _ in range(100):
    lo, hi = K.bootstrap_ci(lambda x: np.median(x) / 0.0135, rng.normal(0, 0.02, 100), B=300, rng=rng)
    covers += lo <= 0 <= hi
res.append(("自助区间：零效应时 95% 区间含 0 的比例 ≥0.85", covers / 100, "≥0.85", covers / 100 >= 0.85))

w = max(len(r[0]) for r in res)
ok = True
for n, g, wv, o in res:
    ok &= o
    print(f"{'PASS' if o else 'FAIL'}  {n:<{w}}  得到={g}  预期={wv}")
print("\n全部演示符合预期" if ok else "\n存在不符合预期的演示")
sys.exit(0 if ok else 1)
