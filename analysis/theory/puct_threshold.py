"""Q5 推导：只污染策略网络的后门，被 KataGo PUCT 搜索冲掉所需的访问数。

公式取自 KataGo master d743f217：
  cpp/search/searchexplorehelpers.cpp:9-53
    U(child) = c(N) * sqrt(N + 0.01) * sigma * P / (1 + n) + Q
    c(N)     = cpuctExploration + cpuctExplorationLog * ln((N + base) / base)
  cpp/program/setup.cpp:556-572 默认值：cpuctExploration = 1.0、cpuctExplorationLog = 0.45、base = 500、
    cpuctUtilityStdevScale = 0.85（非分布式训练时）、cpuctUtilityStdevPrior = 0.40
  sigma = 1 + 0.85 * (父节点效用标准差 / 0.40 - 1)（searchexplorehelpers.cpp:297），本文把它当参数。

理想化（推导档的前提，必须写明）：
  - 只看根节点的两个子节点：后门着法 m（先验 P_m，效用 u - du）与最好着法 b（先验 P_b，效用 u）；其余着法先验 1 - P_m - P_b、效用低到不被访问。
  - 价值网络是对的：每个子节点的 Q 等于其真效用（确定性版本），或真效用加每次访问独立噪声（噪声版本）。
  - 最终着法按访问数最多选（忽略 LCB 选择与 FPU）。
  - 效用只计胜负项：胜率差 dp 对应效用差 du = 2 * dp（winLossUtilityFactor = 1；忽略得分效用项）。

闭式近似：在 n_m = n_b = N/2 处两者 U 相等，
  c(N) * sigma * (P_m - P_b) * sqrt(N) / (1 + N/2) = du
  => N* ≈ (2 * c(N*) * sigma * (P_m - P_b) / du) ** 2   （c 随 N 对数增长，用不动点迭代）
预测：N* 与危害（du）的平方成反比——危害越轻的后门越扛得住搜索。
"""
import math
import random

C0, CLOG, BASE = 1.0, 0.45, 500.0


def c_of(n):
    return C0 + CLOG * math.log((n + BASE) / BASE)


def closed_form(pm, pb, dp, sigma):
    du = 2.0 * dp
    n = 1.0
    for _ in range(200):
        n = (2.0 * c_of(n) * sigma * (pm - pb) / du) ** 2
    return n


def simulate(pm, pb, dp, sigma, noise_sd=0.0, max_n=200000, rng=None):
    """返回最好着法访问数首次超过后门着法的总访问数 N*；超过 max_n 返回 None。"""
    du = 2.0 * dp
    u_b, u_m = 0.0, -du
    n = {"m": 0, "b": 0}
    s = {"m": 0.0, "b": 0.0}
    prior = {"m": pm, "b": pb}
    true_u = {"m": u_m, "b": u_b}
    total = 0
    while total < max_n:
        best_k, best_v = None, -1e18
        for k in ("m", "b"):
            q = s[k] / n[k] if n[k] > 0 else true_u[k]  # 首次访问前用真值当 FPU 的理想化替代
            v = c_of(total) * math.sqrt(total + 0.01) * sigma * prior[k] / (1 + n[k]) + q
            if v > best_v:
                best_k, best_v = k, v
        x = true_u[best_k] + (rng.gauss(0.0, noise_sd) if noise_sd > 0 else 0.0)
        n[best_k] += 1
        s[best_k] += x
        total += 1
        if total >= 2 and n["b"] > n["m"]:
            return total
    return None


def main():
    print("# 确定性模拟 vs 闭式（P_b = 0.05）")
    print("P_m\tsigma\tdp(胜率)\t闭式N*\t模拟N*")
    for pm in (0.5, 0.9):
        for sigma in (0.35, 1.0):
            for dp in (0.01, 0.02, 0.05, 0.10, 0.20, 0.40):
                cf = closed_form(pm, 0.05, dp, sigma)
                sim = simulate(pm, 0.05, dp, sigma)
                print(f"{pm}\t{sigma}\t{dp}\t{cf:.0f}\t{sim if sim is not None else '>200000'}")
    print()
    print("# 噪声版本（每次访问效用噪声 sd=0.3，200 个种子的中位数；P_m=0.9, P_b=0.05, sigma=1.0）")
    print("dp\t中位N*\t10分位\t90分位\t未翻转比例")
    rng = random.Random(20261127)
    for dp in (0.02, 0.05, 0.10, 0.20):
        vals, none = [], 0
        for _ in range(200):
            r = simulate(0.9, 0.05, dp, 1.0, noise_sd=0.3, max_n=60000, rng=rng)
            if r is None:
                none += 1
            else:
                vals.append(r)
        vals.sort()
        if vals:
            q = lambda f: vals[min(len(vals) - 1, int(f * len(vals)))]
            print(f"{dp}\t{q(0.5)}\t{q(0.1)}\t{q(0.9)}\t{none/200:.2f}")
        else:
            print(f"{dp}\t-\t-\t-\t{none/200:.2f}")


if __name__ == "__main__":
    main()
