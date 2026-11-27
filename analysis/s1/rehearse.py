"""S1 判据离线预演（准则 6.2）：在合成数据上同时演示每个判据"能被满足"与"能不被满足"，
并检查在零假设下误判率是否在名义水平附近。

用法：python3 -I analysis/s1/rehearse.py
任一演示与预期不符即以非零码退出。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import criteria as C  # noqa: E402

rng = np.random.default_rng(20261127)
results = []


def expect(name, got, want):
    ok = got == want
    results.append((name, got, want, ok))


# E0
expect("E0 确定性：完全一致", C.e0_determinism(np.arange(200), np.arange(200))[0], True)
b = np.arange(200).copy(); b[:10] = -1
expect("E0 确定性：5% 不一致", C.e0_determinism(np.arange(200), b)[0], False)
expect("E0 循环复现：60% 错着", C.e0_cyclic_reproduction(rng.random(83) < 0.6)[0], True)
expect("E0 循环复现：20% 错着", C.e0_cyclic_reproduction(rng.random(83) < 0.2)[0], False)
expect("E0-4 前向比对：一致", C.e0_torch_engine_match(np.ones(200, bool), np.full(200, 1e-5))[0], True)
expect("E0-4 前向比对：KL 过大", C.e0_torch_engine_match(np.ones(200, bool), np.full(200, 1e-2))[0], False)

# E1a：有判别力 vs 随机分数 vs 样本不足
bad = rng.normal(1.5, 1, 200); ok = rng.normal(0, 1, 2000)
expect("E1a 有判别力（d=1.5）", C.e1a_verdict(bad, ok, rng=rng)["status"], "可辨识")
bad = rng.normal(0, 1, 200); ok = rng.normal(0, 1, 2000)
expect("E1a 随机分数", C.e1a_verdict(bad, ok, rng=rng)["status"], "不可辨识")
expect("E1a 真错不足 30", C.e1a_verdict(rng.normal(2, 1, 20), ok, rng=rng)["status"], "样本不足")

# E1a 零假设下的误判率（应接近 5%）
fa = 0
for _ in range(200):
    v = C.e1a_verdict(rng.normal(0, 1, 100), rng.normal(0, 1, 1000), n_perm=200, rng=rng)
    fa += v["status"] == "可辨识"
null_rate = fa / 200
results.append(("E1a 零假设误判率（名义 0.05，容许 ≤0.10）", round(null_rate, 3), "≤0.10", null_rate <= 0.10))

# E1b
fl = C.e1b_floor({"近": rng.exponential(0.01, 2000), "远": rng.exponential(0.2, 2000)}, cap99=0.3)
expect("E1b 近距离未被淹没", fl["近"][3], False)
expect("E1b 远距离被淹没", fl["远"][3], True)


# E3：注入局部分歧 vs 只有整体漂移
def pair(local_shift=0.0, global_scale=1.0):
    norm = rng.exponential(0.02 * global_scale, 83)
    cyc = rng.exponential(0.02 * global_scale, 83) + local_shift
    return C.e3_statistic(cyc, norm)


controls = [pair() for _ in range(25)]
expect("E3 局部改变（跨事件对 S' 抬高）", C.e3_verdict([pair(0.1), pair(0.1), pair(0.1)], controls)["status"], "通过")

# E3 真有局部改变时的检出率（重复 300 次，应 ≥0.9）
hit = 0
for _ in range(300):
    cs = [pair() for _ in range(25)]
    hit += C.e3_verdict([pair(0.1), pair(0.1), pair(0.1)], cs)["status"] == "通过"
results.append(("E3 真有局部改变时的通过率（应 ≥0.9）", round(hit / 300, 3), "≥0.9", hit / 300 >= 0.9))

# E3 只有整体漂移时的误判率（重复 300 次，应 <0.01）
fa = 0
for _ in range(300):
    cs = [pair() for _ in range(25)]
    fa += C.e3_verdict([pair(0.0, 3.0), pair(0.0, 3.0), pair(0.0, 3.0)], cs)["status"] == "通过"
results.append(("E3 只有整体漂移时的误判率（应 <0.01）", round(fa / 300, 4), "<0.01", fa / 300 < 0.01))
expect("E3 只有整体漂移", C.e3_verdict([pair(0.0, 3.0), pair(0.0, 3.0)], controls)["status"], "不通过")
expect("E3 无局部改变", C.e3_verdict([pair(0.0), pair(0.0), pair(0.0)], controls)["status"], "不通过")
expect("E3 对照不足", C.e3_verdict([pair(0.1)], controls[:10])["status"], "对照不足")

# E3 零假设误判率：跨事件对与对照同分布时，3 对全过的概率应很小（每对约 0.05×0.9）
fa = 0
for _ in range(500):
    cs = [pair() for _ in range(25)]
    fa += C.e3_verdict([pair(), pair(), pair()], cs)["status"] == "通过"
e3_null = fa / 500
results.append(("E3 零假设误判率（3 对全过，应 <0.01）", round(e3_null, 4), "<0.01", e3_null < 0.01))

# G4
expect("G4 区间窄", C.g4_verdict(13000, 13300, 482, 14642)["status"], "通过")
expect("G4 区间宽", C.g4_verdict(4000, 12000, 482, 14642)["status"], "不通过")

width = max(len(r[0]) for r in results)
allok = True
for name, got, want, ok in results:
    allok &= ok
    print(f"{'PASS' if ok else 'FAIL'}  {name:<{width}}  得到={got}  预期={want}")
print("\n全部演示符合预期" if allok else "\n存在不符合预期的演示")
sys.exit(0 if allok else 1)
