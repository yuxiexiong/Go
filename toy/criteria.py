"""玩具实验判定（迷你合同 §4）。只吃汇总读数，吐判定。阈值为草案值，冻结时以合同为准。"""
import numpy as np


def bootstrap_ci(values_fn, data, B=2000, rng=None, q=(0.025, 0.975)):
    rng = rng or np.random.default_rng(0)
    data = np.asarray(data, dtype=float)
    stats = np.array([values_fn(rng.choice(data, size=len(data), replace=True)) for _ in range(B)])
    return float(np.quantile(stats, q[0])), float(np.quantile(stats, q[1]))


def implant_ok(flip_rate_trig, min_flip=0.5):
    """玩具 B：触发局面上只用策略网络时首选被改的比例。"""
    return flip_rate_trig >= min_flip


def strength_ok(agree_bd, agree_twin, tol=0.05):
    """玩具 B：非触发局面上与原网络首选一致率，被植入臂不低于孪生臂减 tol。"""
    return agree_bd >= agree_twin - tol


def shape_verdict(ci_strong, ci_weak, max_width=2.0):
    """小危害档：最强与最弱审计者的 d 的 95% 置信区间。
    有形状 ⇔ 两区间不重叠；平 ⇔ 重叠且两区间宽度都 ≤ max_width；看不出 ⇔ 其余（太吵）。"""
    (a0, a1), (b0, b1) = ci_strong, ci_weak
    if a1 < b0 or b1 < a0:
        return "有形状"
    if (a1 - a0) <= max_width and (b1 - b0) <= max_width:
        return "平"
    return "看不出"


def decide(implant, strength, d_large_by_aud, shape, d_alive=2.0, d_dead=1.0):
    """总判定（合同 §4 判定表）。implant/strength 对玩具 A 传 True（不适用）。"""
    if not implant or not strength:
        return "改植入配方"
    if max(d_large_by_aud.values()) < d_dead:
        return "信号被淹没：改设计"
    if max(d_large_by_aud.values()) >= d_alive and shape in ("有形状", "平"):
        return "课题成立：进入 S1/S2"
    return "边界：报告并讨论"
