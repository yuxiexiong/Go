"""Q5 推导的修订版：在 puct_threshold.py 的理想化上加入 KataGo 的 FPU 与 LCB 选着（2026-11-27 对抗自检第 34 条）。

依据（KataGo master d743f217）：
  - 探索项同 puct_threshold.py（searchexplorehelpers.cpp:9-53, 297；setup.cpp:556-572）。
  - FPU（searchexplorehelpers.cpp:265-320；setup.cpp:577, 660）：未访问子节点的 Q = 父节点平均效用 − 根节点 rootFpuReductionMax(0.1) × sqrt(已访问子节点的策略质量)。
  - LCB 选着（searchresults.cpp:196-240；searchhelpers.cpp:572-612；setup.cpp:706-712）：
      候选 = 访问数 ≥ minVisitPropForLCB(0.15) × 访问数最多者 的子节点；
      LCB = 平均效用 − lcbStdevs(5) × sqrt(效用方差 / n)，方差含一个以效用半径 1.4 为尺度、权重 1/n^2 的先验；
      选 LCB 最高者。
仍未建模：getReducedPlaySelectionWeight 对非最佳子节点访问数的折减、深层树、得分效用、对称采样、批量/虚拟损失。
所以结论仍是"理想化推导"，常数不能直接搬到真实 KataGo；S1.5/S2 用真实引擎测。

输出：在访问预算网格上，最终选中"最好着法 b"的种子比例；对比"按访问数选"与"按 LCB 选"。
"""
import math
import random

C0, CLOG, BASE = 1.0, 0.45, 500.0
FPU_RED = 0.1
LCB_STDEVS, LCB_MINPROP = 5.0, 0.15
UTIL_RADIUS = 1.0 + 0.1 + 0.3


def c_of(n):
    return C0 + CLOG * math.log((n + BASE) / BASE)


def run(pm, pb, dp, sigma, noise_sd, budgets, rng):
    """返回 {预算: (按访问数是否选 b, 按 LCB 是否选 b)}。"""
    du = 2.0 * dp
    true_u = {"m": -du, "b": 0.0}
    prior = {"m": pm, "b": pb}
    n = {"m": 0, "b": 0}
    s = {"m": 0.0, "b": 0.0}
    s2 = {"m": 0.0, "b": 0.0}
    out = {}
    total, parent_sum = 0, 0.0
    want = sorted(budgets)
    wi = 0
    while wi < len(want):
        parent_q = parent_sum / total if total > 0 else 0.0
        visited_mass = sum(prior[k] for k in n if n[k] > 0)
        fpu = parent_q - FPU_RED * math.sqrt(visited_mass)
        best_k, best_v = None, -1e18
        for k in ("m", "b"):
            q = s[k] / n[k] if n[k] > 0 else fpu
            v = c_of(total) * math.sqrt(total + 0.01) * sigma * prior[k] / (1 + n[k]) + q
            if v > best_v:
                best_k, best_v = k, v
        x = true_u[best_k] + (rng.gauss(0.0, noise_sd) if noise_sd > 0 else 0.0)
        n[best_k] += 1
        s[best_k] += x
        s2[best_k] += x * x
        total += 1
        parent_sum += x
        while wi < len(want) and total == want[wi]:
            by_visits = "b" if n["b"] > n["m"] else "m"
            nmax = max(n.values())
            best_lcb, lcb_pick = -1e18, by_visits
            for k in ("m", "b"):
                if n[k] == 0 or n[k] < LCB_MINPROP * nmax:
                    continue
                mean = s[k] / n[k]
                var = max(s2[k] / n[k] - mean * mean, 1e-8)
                prior_w = 1.0 / (n[k] ** 2)
                var = (var * n[k] + (var + UTIL_RADIUS ** 2) * prior_w) / (n[k] + prior_w)
                lcb = mean - LCB_STDEVS * math.sqrt(var / n[k])
                if lcb > best_lcb:
                    best_lcb, lcb_pick = lcb, k
            out[want[wi]] = (by_visits == "b", lcb_pick == "b")
            wi += 1
    return out


def main():
    budgets = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    rng = random.Random(20261127)
    for noise in (0.0, 0.3):
        seeds = 1 if noise == 0 else 200
        print(f"\n# 效用噪声 sd={noise}（{seeds} 个种子）；格子里是 '按访问数选中 b 的比例 / 按 LCB 选中 b 的比例'")
        for pm, sigma in ((0.9, 1.0), (0.9, 0.35), (0.5, 1.0)):
            print(f"\nP_m={pm}, P_b=0.05, sigma={sigma}")
            print("dp\\预算\t" + "\t".join(str(b) for b in budgets))
            for dp in (0.01, 0.02, 0.05, 0.10, 0.20):
                acc = {b: [0, 0] for b in budgets}
                for _ in range(seeds):
                    r = run(pm, 0.05, dp, sigma, noise, budgets, rng)
                    for b in budgets:
                        acc[b][0] += r[b][0]
                        acc[b][1] += r[b][1]
                cells = [f"{acc[b][0]/seeds:.2f}/{acc[b][1]/seeds:.2f}" for b in budgets]
                print(f"{dp}\t" + "\t".join(cells))


if __name__ == "__main__":
    main()
