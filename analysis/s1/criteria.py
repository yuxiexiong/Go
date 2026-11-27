"""S1 合同 v2（docs/CONTRACT_S1_DRAFT_2026-11-27.zh-CN.md）的判读函数。

每个函数只吃读数、吐判定；判读归本模块，跑实验的脚本不自己算判据。
阈值以参数传入，默认值即合同草案值；冻结时以合同为准。
v2 变更（2026-11-27 对抗自检）：E1a 改称"判别力"而非"地板"；E1b 只报告不设闸；
E3 按事件、按同结构对照判，增加盲审版；G4 改为"区间宽度 ≤ 越线区间相邻配置间距"；新增 E1c 真值噪声。
"""
import numpy as np


# ---------- E0 ----------
def e0_determinism(run_a, run_b, min_agree=0.99):
    a, b = np.asarray(run_a), np.asarray(run_b)
    agree = float(np.mean(np.all(a == b, axis=1))) if a.ndim > 1 else float(np.mean(a == b))
    return agree >= min_agree, agree


def e0_cyclic_reproduction(wrong_first_choice, min_frac=0.5):
    frac = float(np.mean(np.asarray(wrong_first_choice, dtype=bool)))
    return frac >= min_frac, frac


def e0_torch_engine_match(top1_agree, kl, min_top1=0.99, max_median_kl=1e-3):
    t = float(np.mean(np.asarray(top1_agree, dtype=bool)))
    m = float(np.median(np.asarray(kl)))
    return (t >= min_top1 and m <= max_median_kl), t, m


# ---------- E1a：天然错误判别力（局面级、已知标签） ----------
def tpr_at_fpr(score_bad, score_ok, fpr):
    thr = np.quantile(np.asarray(score_ok, dtype=float), 1.0 - fpr)
    return float(np.mean(np.asarray(score_bad, dtype=float) > thr))


def e1a_verdict(score_bad, score_ok, fpr=0.05, n_perm=1000, min_bad=30, rng=None):
    """status ∈ {样本不足, 有判别力, 无判别力}。

    零分布：在 (bad ∪ ok) 上置换标签重算 TPR。有判别力 ⇔ 实测 TPR 高于零分布 95 分位
    （等价于检验分数比随机好；不是"地板"——误报率已被 fpr 固定）。
    """
    rng = rng or np.random.default_rng(0)
    score_bad = np.asarray(score_bad, dtype=float)
    score_ok = np.asarray(score_ok, dtype=float)
    if len(score_bad) < min_bad:
        return {"status": "样本不足", "tpr": None, "null95": None}
    tpr = tpr_at_fpr(score_bad, score_ok, fpr)
    pool = np.concatenate([score_bad, score_ok])
    nb = len(score_bad)
    null = np.empty(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(pool)
        null[i] = tpr_at_fpr(perm[:nb], perm[nb:], fpr)
    q95 = float(np.quantile(null, 0.95))
    return {"status": "有判别力" if tpr > q95 else "无判别力", "tpr": tpr, "null95": q95}


# ---------- E1b：参照漂移分布（只报告） ----------
def e1b_report(divergences_by_bin):
    """{距离箱: 每局面分歧数组} → {距离箱: (q50, q95, q99)}。不设闸门。"""
    return {k: tuple(float(np.quantile(np.asarray(v, dtype=float), q)) for q in (0.5, 0.95, 0.99))
            for k, v in divergences_by_bin.items()}


# ---------- E1c：真值噪声 ----------
def e1c_epsilons(symmetry_sd, k_ok=2.0, k_bad=5.0):
    """由真值配置在对称变换下的胜率标准差（每局面一个）定 ε_ok、ε_bad。"""
    med = float(np.median(np.asarray(symmetry_sd, dtype=float)))
    return {"noise_median": med, "eps_ok": k_ok * med, "eps_bad": k_bad * med}


# ---------- E3：天然阳性对照 ----------
def e3_statistic(div_cyc, div_norm, eps=1e-12):
    """S' = (循环局面分歧中位数 − 普通局面分歧中位数) / 普通局面分歧 MAD。"""
    div_cyc = np.asarray(div_cyc, dtype=float)
    div_norm = np.asarray(div_norm, dtype=float)
    mad = float(np.median(np.abs(div_norm - np.median(div_norm))))
    return float((np.median(div_cyc) - np.median(div_norm)) / (mad + eps))


def e3_verdict(event_points, controls, min_controls=20):
    """已知位置版。event_points：同一事件的被审点 S' 列表；controls：同结构、按训练样本距离匹配的对照 S'。

    通过 ⇔ 该事件所有被审点的 S' 都高于对照 95 分位。
    """
    if len(controls) < min_controls:
        return {"status": "对照不足", "n_controls": len(controls)}
    s95 = float(np.quantile(np.asarray(controls, dtype=float), 0.95))
    ok = all(float(x) > s95 for x in event_points)
    return {"status": "通过" if ok else "不通过", "S95": s95}


def ranking_auc(div_all, is_cyc):
    """把局面按分歧降序排，P_cyc 的 AUC（Mann–Whitney 形式）。"""
    div_all = np.asarray(div_all, dtype=float)
    is_cyc = np.asarray(is_cyc, dtype=bool)
    pos, neg = div_all[is_cyc], div_all[~is_cyc]
    ranks = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1
    r_pos = ranks[: len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def e3_blind_verdict(event_aucs, control_aucs, min_controls=20):
    """盲审版。通过 ⇔ 该事件所有被审点的排序 AUC 高于对照 AUC 的 95 分位。"""
    if len(control_aucs) < min_controls:
        return {"status": "对照不足", "n_controls": len(control_aucs)}
    a95 = float(np.quantile(np.asarray(control_aucs, dtype=float), 0.95))
    ok = all(float(x) > a95 for x in event_aucs)
    return {"status": "通过" if ok else "不通过", "AUC95": a95}


# ---------- G4：人类锚点 ----------
def g4_verdict(anchor_lo, anchor_hi, crossing_spacing):
    """人类锚点 95% 区间宽度 ≤ 越线区间内相邻配置的强度间距（同一 Elo 单位）。"""
    width = anchor_hi - anchor_lo
    return {"status": "通过" if width <= crossing_spacing else "不通过", "width": float(width)}
