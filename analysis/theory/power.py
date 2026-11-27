"""检出功效：一个审计者在 M 个局面上审一个模型，能否在模型级误报率 alpha_m 下判出后门。

模型（推导档；分布形状是占位，S1 用实测地板替换）：
  - 审计者对每个局面给一个异常分数。干净局面的分数 ~ N(0,1)（"地板"，即干净谱系漂移，已标准化）。
  - 触发局面的分数 ~ N(d, 1)。d 就是"信号地板比"在标准化单位下的版本。
  - 审计者不知道触发是什么，按部署分布抽 M 个局面，其中触发局面比例为 rho（触发暴露量 = rho * M）。
  - 局面级阈值取干净分数的 1 - alpha_p 分位；模型级统计量 = 被标记的局面数 K。
  - 模型级判定：K 超过"干净模型下 K ~ Binomial(M, alpha_p)"的 1 - alpha_m 分位即判有后门。
输出：给定 (d, rho, M) 时的模型级检出率；以及检出率达到 0.8 所需的最小 M。
"""
import math


def phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def z_of(p):  # 标准正态上分位点（二分）
    lo, hi = -10.0, 10.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if 1 - phi(mid) > p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def binom_sf_threshold(m, p, alpha):
    """最小 k，使 P(Binomial(m,p) > k) <= alpha。

    m <= 20000 时在对数空间逐项累加（避免 (1-p)^m 下溢；2026-11-27 首版在此下溢，
    把大 m 的阈值错算成 m，已修正）；更大的 m 用正态近似。
    """
    if m > 20000:
        mu, sd = m * p, math.sqrt(m * p * (1 - p))
        return math.ceil(mu + z_of(alpha) * sd)
    log_p, log_q = math.log(p), math.log(1 - p)
    cdf = 0.0
    for k in range(m + 1):
        log_pmf = (math.lgamma(m + 1) - math.lgamma(k + 1) - math.lgamma(m - k + 1)
                   + k * log_p + (m - k) * log_q)
        cdf += math.exp(log_pmf)
        if 1 - cdf <= alpha:
            return k
    return m


def model_tpr(d, rho, m, alpha_p=0.01, alpha_m=0.05):
    z = z_of(alpha_p)
    k_star = binom_sf_threshold(m, alpha_p, alpha_m)
    p_flag_trig = 1 - phi(z - d)
    p_flag = rho * p_flag_trig + (1 - rho) * alpha_p
    # P(Binomial(m, p_flag) > k_star)，m 大时用正态近似加连续性修正
    mu, var = m * p_flag, m * p_flag * (1 - p_flag)
    if var <= 0:
        return 1.0 if mu > k_star else 0.0
    return 1 - phi((k_star + 0.5 - mu) / math.sqrt(var))


def min_m(d, rho, target=0.8, alpha_p=0.01, alpha_m=0.05, m_max=10**7):
    m = 10
    while m <= m_max:
        if model_tpr(d, rho, m, alpha_p, alpha_m) >= target:
            return m
        m = int(m * 1.25) + 1
    return None


def main():
    print("# 模型级检出率达到 0.8 所需的局面数 M（alpha_p=0.01, alpha_m=0.05）")
    print("信号d\\触发比例rho\t" + "\t".join(str(r) for r in (0.1, 0.01, 0.001, 0.0001)))
    for d in (0.5, 1.0, 2.0, 3.0, 5.0):
        row = []
        for rho in (0.1, 0.01, 0.001, 0.0001):
            mm = min_m(d, rho)
            row.append(f"{mm} (暴露{mm*rho:.1f})" if mm else ">1e7")
        print(f"{d}\t" + "\t".join(row))
    print()
    print("# 固定暴露量 rho*M 时，信号再强检出率也有上限（rho=0.001）")
    print("暴露量\td=3\td=5\td=10")
    for exp in (1, 3, 10, 30, 100):
        m = int(exp / 0.001)
        print(f"{exp}\t" + "\t".join(f"{model_tpr(d, 0.001, m):.2f}" for d in (3, 5, 10)))


if __name__ == "__main__":
    main()
