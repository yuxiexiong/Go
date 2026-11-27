"""检出功效的修订版（2026-11-27 对抗自检第 13、59 条）：比较不同检验统计量，并同时固定局面数 M 与暴露量 E。

power.py 只用了"超过 1% 局部阈值的局面计数"这一种检验，功效被干净局面的计数噪声卡住，
得出"检出主要由暴露量决定"——那是检验选择的产物。本脚本比较：
  - count(α_p)：局部阈值 α_p ∈ {1e-2, 1e-3, 1e-4} 的计数检验（解析，二项分布）；
  - max：最大分数检验（解析：零假设下 P(max ≤ t) = Φ(t)^M；备择下 Φ(t)^(M−E) Φ(t−d)^E）；
  - HC：Donoho–Jin higher criticism（模拟，零分布由模拟标定）；
  - LR：已知 d 与 E/M 的似然比检验（模拟），作为功效上界的参照。
模型（占位，推导档）：干净分数 ~ N(0,1)，触发分数 ~ N(d,1)；M 个局面中恰有 E 个触发；模型级误报率 α_m = 0.05。
"""
import math

import numpy as np

ALPHA_M = 0.05


def phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def zq(p):  # 上分位点
    lo, hi = -40.0, 40.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if 1 - phi(mid) > p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def log_binom_pmf(k, m, p):
    return (math.lgamma(m + 1) - math.lgamma(k + 1) - math.lgamma(m - k + 1)
            + k * math.log(p) + (m - k) * math.log1p(-p))


def count_power(d, e, m, ap):
    # 阈值：最小 k* 使 P(Bin(m, ap) > k*) ≤ α_m
    cdf, kstar = 0.0, m
    for k in range(m + 1):
        cdf += math.exp(log_binom_pmf(k, m, ap))
        if 1 - cdf <= ALPHA_M:
            kstar = k
            break
    p_trig = 1 - phi(zq(ap) - d)
    # 备择：Bin(m−e, ap) + Bin(e, p_trig) > k*，卷积
    clean = np.array([math.exp(log_binom_pmf(k, m - e, ap)) for k in range(0, min(m - e, kstar + 1) + 1)])
    trig = np.array([math.exp(log_binom_pmf(k, e, p_trig)) for k in range(e + 1)]) if 0 < p_trig < 1 else (
        np.eye(1, e + 1, e if p_trig >= 1 else 0)[0])
    p_le = 0.0
    for kt, pt in enumerate(trig):
        lim = kstar - kt
        if lim >= 0:
            p_le += pt * clean[:lim + 1].sum()
    return 1 - p_le


def max_power(d, e, m):
    # 零假设下 P(max > t) = α_m  →  Φ(t) = (1−α_m)^(1/m)
    target = (1 - ALPHA_M) ** (1.0 / m)
    lo, hi = -10.0, 40.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if phi(mid) < target:
            lo = mid
        else:
            hi = mid
    t = (lo + hi) / 2
    return 1 - (phi(t) ** (m - e)) * (phi(t - d) ** e)


def hc_stat(x, alpha0=0.1):
    """Higher criticism，只在最小的 alpha0 比例 p 值上取最大（Donoho–Jin 常用设定）。"""
    n = len(x)
    k = max(10, int(alpha0 * n))
    top = np.sort(x)[::-1][:k]
    p = np.array([0.5 * math.erfc(v / math.sqrt(2.0)) for v in top])
    i = np.arange(1, k + 1)
    hc = np.sqrt(n) * (i / n - p) / np.sqrt(p * (1 - p) + 1e-300)
    return float(np.max(hc))


def lr_stat(x, d, frac):
    return float(np.sum(np.log1p(frac * (np.exp(d * x - d * d / 2) - 1))))


def sim_power(d, e, m, n_null=400, n_alt=400, seed=0):
    rng = np.random.default_rng(seed)
    frac = e / m
    null_hc, null_lr, alt_hc, alt_lr = [], [], [], []
    for _ in range(n_null):
        x = rng.standard_normal(m)
        null_hc.append(hc_stat(x)); null_lr.append(lr_stat(x, d, frac))
    for _ in range(n_alt):
        x = rng.standard_normal(m); x[:e] += d
        alt_hc.append(hc_stat(x)); alt_lr.append(lr_stat(x, d, frac))
    thr_hc = np.quantile(null_hc, 1 - ALPHA_M); thr_lr = np.quantile(null_lr, 1 - ALPHA_M)
    return float(np.mean(np.array(alt_hc) > thr_hc)), float(np.mean(np.array(alt_lr) > thr_lr))


def main():
    print("# 模型级检出率（α_m=0.05）。列：count@1e-2 / count@1e-3 / count@1e-4 / max / HC / LR")
    for m in (1000, 10000):
        print(f"\n## M = {m}")
        print("d\tE\tc1e-2\tc1e-3\tc1e-4\tmax\tHC\tLR")
        for d in (1.0, 2.0, 3.0, 5.0):
            for e in (1, 3, 10, 30, 100):
                if e >= m:
                    continue
                c = [count_power(d, e, m, ap) for ap in (1e-2, 1e-3, 1e-4)]
                mx = max_power(d, e, m)
                hc, lr = sim_power(d, e, m, n_null=300 if m > 5000 else 500, n_alt=300 if m > 5000 else 500,
                                   seed=int(d * 100 + e))
                print(f"{d}\t{e}\t" + "\t".join(f"{v:.2f}" for v in (*c, mx, hc, lr)))
    print("\n## M = 100000（只算解析的 count 与 max）")
    print("d\tE\tc1e-2\tc1e-3\tc1e-4\tmax")
    for d in (2.0, 3.0, 5.0):
        for e in (1, 3, 10, 30, 100):
            c = [count_power(d, e, 100000, ap) for ap in (1e-2, 1e-3, 1e-4)]
            print(f"{d}\t{e}\t" + "\t".join(f"{v:.2f}" for v in (*c, max_power(d, e, 100000))))


if __name__ == "__main__":
    main()
