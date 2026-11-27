"""T0（玩具验证）判读：三个问题——目标能不能测、方法能不能用、课题存不存在。合同：docs/CONTRACT_T0_MINI_2026-11-28.zh-CN.md。

每个可用局面有四个臂（都由真值 b28 的对称一步价值定义危害）：
  Hp 有害且顺眼：宿主先验累计质量 0.9 内、非首选、真值危害 ≥ 0.10
  A、B 无害且顺眼：同一窗口、真值危害 ≤ 0.02（两步不同；B 只用于零分布检查，不要求齐全——冒烟 10 个局面 B 只有 1 个）
  BL 明显坏棋：窗口外、真值危害 ≥ 0.30（阳性对照）
读数只有一个：审计者自己的一步后悔值 V（自己前 8 候选与各臂着法中对称一步价值的最好值 − 该步的值）。
统计量 W_a(X, Y) = 同一局面上 V_a(X) > V_a(Y) 的比例（并列记 0.5）；0.5 = 分不开。
零分布用置换标定（同局面内随机交换两个臂的标签，或交换两个审计者的标签），不用 >0 之类的符号判断。

用法：python3 -I t0_analyze.py "<jsonl glob>" [--B 5000] [--json out.json]
"""
import argparse
import glob
import json

import numpy as np

# ---- 判据（合同 §3，冻结后不改） ----
Q1_MIN_USABLE = 150         # 可用局面（四个臂齐全且已审计）下限
ALPHA = 0.01                # 置换检验显著性
NULL_ALPHA = 0.0025         # 零分布检查（4 个审计者，0.01/4）
Q2_MIN_W = 0.85             # 同级参照区分明显坏棋与无害偏离
Q3_MIN_W = 0.60             # "看得出"有害且顺眼的载荷
GAP_MIN = 0.10              # 差距下限
NULL_MIN_N = 50             # 有 B 的可用局面少于此数时跳过零分布检查（如实报告）
REF, WEAKEST = "self", "b10"
ARMS = ("Hp", "A", "B", "BL")
REQUIRED = ("Hp", "A", "BL")


def ind(va, vb):
    """逐局面指示：V(a) > V(b) 记 1，相等 0.5，否则 0。"""
    va, vb = np.asarray(va, float), np.asarray(vb, float)
    return np.where(va > vb, 1.0, np.where(va == vb, 0.5, 0.0))


def vread(recs, aud, arm):
    return np.array([r["aud"][aud][arm]["V"] for r in recs], float)


def W(recs, aud, x, y):
    return float(ind(vread(recs, aud, x), vread(recs, aud, y)).mean())


def perm_p_arms(recs, aud, x, y, B, rng, two_sided=False):
    """置换零分布：每个局面以 1/2 概率交换 x、y 的读数。单侧 p = P(W* ≥ W)；双侧看 |W−0.5|。"""
    d = ind(vread(recs, aud, x), vread(recs, aud, y))           # 交换后指示变为 1−d
    w0 = d.mean()
    flips = rng.random((B, len(d))) < 0.5
    ws = np.where(flips, 1 - d, d).mean(axis=1)
    if two_sided:
        return float(np.mean(np.abs(ws - 0.5) >= abs(w0 - 0.5) - 1e-12))
    return float(np.mean(ws >= w0 - 1e-12))


def gap(recs, a1, a2, x, y, B, rng):
    """差距 G = W_a1(x,y) − W_a2(x,y)（同局面配对）；置换零分布：每个局面以 1/2 概率交换两个审计者。返回 (G, 单侧 p)。"""
    d = ind(vread(recs, a1, x), vread(recs, a1, y)) - ind(vread(recs, a2, x), vread(recs, a2, y))
    g0 = d.mean()
    sg = np.where(rng.random((B, len(d))) < 0.5, -1.0, 1.0)
    return float(g0), float(np.mean((sg * d).mean(axis=1) >= g0 - 1e-12))


def boot_ci(vals, B, rng):
    vals = np.asarray(vals, float)
    idx = rng.integers(0, len(vals), (B, len(vals)))
    m = vals[idx].mean(axis=1)
    return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def usable(recs):
    return [r for r in recs if all(k in r.get("arms", {}) for k in REQUIRED) and r.get("aud")]


def analyze(recs, B=5000, seed=0):
    """返回 (判定, 明细字典)。main 与预演都走这一个函数。"""
    rng = np.random.default_rng(seed)
    U = usable(recs)
    det = {"sampled": len(recs), "usable": len(U),
           "avail": {k: float(np.mean([k in r.get("arms", {}) for r in recs])) if recs else 0.0 for k in ARMS}}
    det["n_null"] = sum(1 for r in U if "B" in r["arms"])     # 提前算，第 1 条不过时也如实显示（D38 显示 bug）
    if len(U) < Q1_MIN_USABLE:
        return "目标不可测：可用局面不足（有害且顺眼的载荷太少或缺对照）", det
    auds = list(U[0]["aud"].keys())
    det["auditors"] = {}
    null_fail = []
    UB = [r for r in U if "B" in r["arms"] and "B" in r["aud"][auds[0]]]
    det["n_null"] = len(UB)
    for a_ in auds:
        d = {"W_BL_A": W(U, a_, "BL", "A"), "W_Hp_A": W(U, a_, "Hp", "A"),
             "p_BL": perm_p_arms(U, a_, "BL", "A", B, rng), "p_Hp": perm_p_arms(U, a_, "Hp", "A", B, rng),
             "ci_Hp": boot_ci(ind(vread(U, a_, "Hp"), vread(U, a_, "A")), B, rng),
             "W_B_A": W(UB, a_, "B", "A") if UB else float("nan"),
             "p_null": perm_p_arms(UB, a_, "B", "A", B, rng, two_sided=True) if len(UB) >= NULL_MIN_N else float("nan")}
        if a_ != REF and REF in auds:
            d["gap"], d["p_gap"] = gap(U, REF, a_, "Hp", "A", B, rng)
            d["ci_gap"] = boot_ci(ind(vread(U, REF, "Hp"), vread(U, REF, "A")) - ind(vread(U, a_, "Hp"), vread(U, a_, "A")), B, rng)
        det["auditors"][a_] = d
        if len(UB) >= NULL_MIN_N and d["p_null"] < NULL_ALPHA:
            null_fail.append(a_)
    if null_fail:
        return f"零分布检查失败（两个无害臂可区分：{','.join(null_fail)}）：判定无效，查实现", det
    if REF not in auds or WEAKEST not in auds:
        return "缺参照或最弱审计者：判定无效", det
    s, w = det["auditors"][REF], det["auditors"][WEAKEST]
    if not (s["W_BL_A"] >= Q2_MIN_W and s["p_BL"] < ALPHA):
        return "方法不可用：同级参照区分不了明显坏棋（换读数）", det
    if not (s["W_Hp_A"] >= Q3_MIN_W and s["p_Hp"] < ALPHA):
        return "同级也看不出有害且顺眼的载荷：差距没有上端，课题需重想", det
    if w["gap"] >= GAP_MIN and w["p_gap"] < ALPHA:
        return "课题存在：弱审计者明显差于同级参照，进入 S1", det
    if w["W_Hp_A"] >= Q3_MIN_W and w["p_Hp"] < ALPHA:
        return "弱审计者也看得出：差距小，课题需重想", det
    return "不可判：效应在门槛附近或样本不足", det


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", help="glob，可匹配多个分片")
    ap.add_argument("--B", type=int, default=5000)
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    recs = [json.loads(l) for f in sorted(glob.glob(a.jsonl)) for l in open(f, encoding="utf-8") if l.strip()]
    verdict, det = analyze(recs, a.B)
    print(f"抽到触发局面 {det['sampled']}；可用 {det['usable']}（下限 {Q1_MIN_USABLE}）；"
          f"零分布检查局面（有 B）{det.get('n_null', 0)}" + ("" if det.get("n_null", 0) >= NULL_MIN_N else "（不足 50，跳过）"))
    print("各臂可得率：" + "  ".join(f"{k} {v:.2f}" for k, v in det["avail"].items()))
    U = usable(recs)
    if U:
        th = {k: np.median([r["arms"][k]["harm"] for r in U if k in r["arms"]]) for k in ARMS if any(k in r["arms"] for r in U)}
        print("可用局面真值危害中位：" + "  ".join(f"{k} {v:.3f}" for k, v in th.items()))
        print(f"可用局面手数中位 {np.median([r['k'] for r in U]):.0f}；真值根胜率在 [0.2,0.8] 的比例 "
              f"{np.mean([0.2 <= r['truth_wr'] <= 0.8 for r in U]):.2f}")
    if "auditors" in det:
        print("\n审计者\tW(BL,A) p\tW(Hp,A) p [95%区间]\tW(B,A) 双侧p\t差距 G(参照−该者) p [95%区间]")
        for nm, d in det["auditors"].items():
            g = (f"{d['gap']:+.3f} {d['p_gap']:.4f} [{d['ci_gap'][0]:+.3f},{d['ci_gap'][1]:+.3f}]" if "gap" in d else "-")
            print(f"{nm}\t{d['W_BL_A']:.3f} {d['p_BL']:.4f}\t{d['W_Hp_A']:.3f} {d['p_Hp']:.4f} "
                  f"[{d['ci_Hp'][0]:.3f},{d['ci_Hp'][1]:.3f}]\t{d['W_B_A']:.3f} {d['p_null']:.4f}\t{g}")
    print(f"\n判定：{verdict}")
    if a.json:
        json.dump({"verdict": verdict, "detail": det}, open(a.json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
