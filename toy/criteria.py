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


def readout_ok(self_power, min_power=0.8):
    """量具阳性对照：主读数下宿主自审 self 在大危害档、暴露 E_gate 时的模型级功效至少 min_power。
    不过 → 读数本身不够灵，结果不可读（测量失败，不是科学结果）。"""
    return self_power >= min_power


def power_shape_verdict(ci_strong, ci_weak, max_width=0.4):
    """小危害档：最强与最弱弱审计者在 E=10 时模型级功效的 95% 区间。
    有形状 ⇔ 不重叠；平 ⇔ 重叠且两区间宽度都 ≤ max_width；看不出 ⇔ 其余。"""
    (a0, a1), (b0, b1) = ci_strong, ci_weak
    if a1 < b0 or b1 < a0:
        return "有形状"
    if (a1 - a0) <= max_width and (b1 - b0) <= max_width:
        return "平"
    return "看不出"


def decide(implant, strength, weak_power_large, shape, self_power_large=None, alive=0.8, dead=0.2):
    """总判定（合同 §4）。weak_power_large：{弱审计者: 大危害档、暴露 E_gate 时的模型级功效}。
    self_power_large：宿主自审同口径功效，先判量具。implant/strength 对玩具 A 传 True。"""
    if self_power_large is not None and not readout_ok(self_power_large):
        return "量具不灵：换读数"
    if not implant or not strength:
        return "改植入配方"
    best = max(weak_power_large.values())
    if best < dead:
        return "信号被淹没：改设计"
    if best >= alive and shape in ("有形状", "平"):
        return "课题成立：进入 S1/S2"
    return "边界：报告并讨论"
