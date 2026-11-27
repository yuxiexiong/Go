"""E2 人类锚点：每对配置下多少局，锚点区间才够窄（推导档：Bradley–Terry / Elo 的误差传播）。

单对：胜率 p 附近，Elo 差 = 400 * log10(p / (1 - p))，
  d(Elo)/dp = 400 / (ln10 * p * (1 - p))，SE(p) = sqrt(p(1-p)/n)
  => SE(Elo) = 400 / (ln10 * sqrt(n * p * (1 - p)))
链式锚定 k 段（每段独立）：SE_链 = SE_单 * sqrt(k)。
判据 G4 用 95% 区间半宽 = 1.96 * SE_链。
"""
import math


def se_pair(n, p=0.5):
    return 400.0 / (math.log(10) * math.sqrt(n * p * (1 - p)))


def games_for_halfwidth(hw, k, p=0.5):
    # 1.96 * se_pair(n) * sqrt(k) <= hw
    n = (1.96 * 400.0 / (math.log(10) * hw)) ** 2 * k / (p * (1 - p))
    return math.ceil(n)


def main():
    print("# 单对 SE(Elo)（p=0.5 与 p=0.8）")
    for n in (50, 100, 200, 400):
        print(f"n={n}\tSE@0.5={se_pair(n):.1f}\tSE@0.8={se_pair(n, 0.8):.1f}")
    print()
    print("# 95% 半宽目标 vs 链长 k 所需每对局数（p=0.5；p=0.8 时乘 1.56）")
    print("半宽\tk=1\tk=2\tk=4\tk=6")
    for hw in (25, 50, 75, 100):
        print(f"{hw}\t" + "\t".join(str(games_for_halfwidth(hw, k)) for k in (1, 2, 4, 6)))


if __name__ == "__main__":
    main()
