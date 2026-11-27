"""S1 合同（docs/CONTRACT_S1_DRAFT_2026-11-27.zh-CN.md）的判读函数。

每个函数只吃读数、吐判定；不读原始序列以外的东西。判读归本模块，跑实验的脚本不自己算判据。
所有阈值以参数传入，默认值即合同草案值；冻结时以合同为准。
"""
import numpy as np


# ---------- E0 ----------
def e0_determinism(run_a, run_b, min_agree=0.99):
    """两次运行的首选着法（或策略向量）一致率。返回 (通过?, 一致率)。"""
    a, b = np.asarray(run_a), np.asarray(run_b)
    if a.ndim > 1:
        agree = float(np.mean(np.all(a == b, axis=1)))
    else:
        agree = float(np.mean(a == b))
    return agree >= min_agree, agree


def e0_cyclic_reproduction(wrong_first_choice, min_frac=0.5):
    """cp505 在 WRONG 标注局面上首选错着的比例。"""
    frac = float(np.mean(np.asarray(wrong_first_choice, dtype=bool)))
    return frac >= min_frac, frac


def e0_torch_engine_match(top1_agree, kl, min_top1=0.99, max_median_kl=1e-3):
    t = float(np.mean(np.asarray(top1_agree, dtype=bool)))
    m = float(np.median(np.asarray(kl)))
    return (t >= min_top1 and m <= max_median_kl), t, m


# ---------- E1a：判官式审计的天然错误检出 ----------
def tpr_at_fpr(score_bad, score_ok, fpr):
    score_ok = np.asarray(score_ok, dtype=float)
    score_bad = np.asarray(score_bad, dtype=float)
    thr = np.quantile(score_ok, 1.0 - fpr)
    return float(np.mean(score_bad > thr))


def e1a_verdict(score_bad, score_ok, fpr=0.05, n_perm=1000, min_bad=30, rng=None):
    """返回 dict：status ∈ {样本不足, 可辨识, 不可辨识}，tpr，零分布 95 分位。

    零分布：在 (bad ∪ ok) 上随机置换标签，重算 TPR。可辨识 ⇔ 实测 TPR 高于零分布 95 分位。
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
    return {"status": "可辨识" if tpr > q95 else "不可辨识", "tpr": tpr, "null95": q95}


# ---------- E1b：参照漂移地板 ----------
def e1b_floor(divergences_by_bin, cap99):
    """divergences_by_bin: {d箱: 每局面分歧数组}。返回 {d箱: (q50,q95,q99, 是否被地板淹没)}。"""
    out = {}
    for k, v in divergences_by_bin.items():
        v = np.asarray(v, dtype=float)
        q50, q95, q99 = (float(np.quantile(v, q)) for q in (0.5, 0.95, 0.99))
        out[k] = (q50, q95, q99, q99 > cap99)
    return out


# ---------- E3：天然阳性对照 ----------
def e3_statistic(div_cyc, div_norm, eps=1e-12):
    """S' = (循环局面分歧中位数 − 普通局面分歧中位数) / 普通局面分歧的 MAD。

    按本对普通局面的离散度标准化，整体漂移同时放大分子与分母，S' 不受其影响。
    （2026-11-27 预演前的草案用未标准化的 S 加"普通局面落在对照 5–95 分位内"的附加条件；
    预演显示附加条件在真有局部改变时约 27% 误判不通过（0.9^3），已替换，见 DECISIONS D14。）
    """
    div_cyc = np.asarray(div_cyc, dtype=float)
    div_norm = np.asarray(div_norm, dtype=float)
    mad = float(np.median(np.abs(div_norm - np.median(div_norm))))
    return float((np.median(div_cyc) - np.median(div_norm)) / (mad + eps))


def e3_verdict(straddle, controls, min_controls=20):
    """straddle: 跨事件对的 S' 列表；controls: 同结构不跨事件对的 S' 列表。

    通过 ⇔ 每个跨事件对的 S' 都高于对照 S' 的 95 分位。
    """
    if len(controls) < min_controls:
        return {"status": "对照不足", "n_controls": len(controls)}
    s95 = float(np.quantile(np.asarray(controls, dtype=float), 0.95))
    per = [{"S'": float(x), "高于对照95分位": float(x) > s95} for x in straddle]
    ok = all(p["高于对照95分位"] for p in per)
    return {"status": "通过" if ok else "不通过", "S95": s95, "pairs": per}


# ---------- G4：人类锚点 ----------
def g4_verdict(anchor_lo, anchor_hi, ladder_lo, ladder_hi, max_frac=0.5):
    """人类锚点 95% 区间映射到阶梯后的覆盖比例。"""
    frac = (anchor_hi - anchor_lo) / (ladder_hi - ladder_lo)
    return {"status": "通过" if frac <= max_frac else "不通过", "frac": float(frac)}
