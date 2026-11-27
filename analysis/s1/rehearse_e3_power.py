"""E3 判据的功效曲线（不作门槛，只供合同定值参考）。用法：python3 -I analysis/s1/rehearse_e3_power.py"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import criteria as C  # noqa: E402

rng = np.random.default_rng(7)


def pair(shift):
    norm = rng.exponential(0.02, 83)
    cyc = rng.exponential(0.02, 83) + shift
    return C.e3_statistic(cyc, norm)


print("局部偏移/普通局面分歧中位数\t3对全过的比例(300次)")
for rel in (0.25, 0.5, 1.0, 1.5, 2.0, 3.0):
    shift = rel * 0.02 * np.log(2)  # 指数分布中位数 = scale*ln2
    hit = 0
    for _ in range(300):
        cs = [pair(0.0) for _ in range(25)]
        hit += C.e3_verdict([pair(shift) for _ in range(3)], cs)["status"] == "通过"
    print(f"{rel}\t{hit/300:.2f}")
