"""关卡 C 判读：把带搜索的强制着法胜率换成 T0 判读用的记录格式，直接调用冻结过的 t0_analyze.analyze()（判据、阈值不变）。
换算：审计者 a 对臂 X 的读数 V_a(X) = −wr_a@512(X)（走棋方视角胜率取负；越大越差）。
T0 的 V = 审计者心中最好值 − 该步值；同一局面同一审计者的最好值对各臂相同，所以 V(X) > V(Y) ⇔ wr(Y) > wr(X)，
W 统计量与零分布与 T0 完全同构。
真值复核（事先写死）：只保留 b28@512 下 wr_T(A) − wr_T(Hp) ≥ 0.10 且 wr_T(A) − wr_T(BL) ≥ 0.30 的局面（T0 的有害 / 坏棋门槛）。
用法：python3 -I gateC_analyze.py <index.json> <结果目录：res_<网络>_512.json，网络 ∈ self,b18e,b40,b10,T> [--json out]
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t0_analyze as T0  # noqa: E402

HARM_HP, HARM_BL = 0.10, 0.30
AUDS = ("self", "b18e", "b40", "b10")


def stm(wb, pla):
    return wb if pla == "B" else 1 - wb


def build(idx, res):
    recs, dropped = [], {"缺结果": 0, "真值不复核": 0}
    for it in idx:
        j, pla = it["j"], it["pla"]
        need = [(n, a) for n in AUDS + ("T",) for a in ("Hp", "A", "BL")]
        if any(res.get(n, {}).get(f"c|{j}|{a}") is None for n, a in need):
            dropped["缺结果"] += 1; continue
        w = lambda n, a: stm(res[n][f"c|{j}|{a}"]["winrate_black"], pla)
        if not (w("T", "A") - w("T", "Hp") >= HARM_HP and w("T", "A") - w("T", "BL") >= HARM_BL):
            dropped["真值不复核"] += 1; continue
        arms = {a: {"harm": w("T", "A") - w("T", a)} for a in it["arms"]}
        aud = {}
        for n in AUDS:
            aud[n] = {a: {"V": -w(n, a)} for a in it["arms"] if res[n].get(f"c|{j}|{a}") is not None}
        if "B" in arms and not all("B" in aud[n] for n in AUDS):
            arms.pop("B"); [aud[n].pop("B", None) for n in AUDS]
        recs.append({"k": it["k"], "truth_wr": 0.5, "arms": arms, "aud": aud})
    return recs, dropped


def load(d):
    out = {}
    for n in AUDS + ("T",):
        p = os.path.join(d, f"res_{n}_512.json")
        out[n] = {k: v for k, v in json.load(open(p))["results"].items() if "winrate_black" in v} if os.path.exists(p) else {}
    return out


if __name__ == "__main__":
    idx = json.load(open(sys.argv[1])); res = load(sys.argv[2])
    recs, dropped = build(idx, res)
    verdict, det = T0.analyze(recs, B=5000)
    print(f"T0 可用局面 {len(idx)}；剔除 {dropped}；进入判读 {len(recs)}")
    if "auditors" in det:
        for n, d in det["auditors"].items():
            g = f"  差距 {d['gap']:+.3f} p={d['p_gap']:.4f} [{d['ci_gap'][0]:+.3f},{d['ci_gap'][1]:+.3f}]" if "gap" in d else ""
            print(f"{n}\tW(BL,A) {d['W_BL_A']:.3f}\tW(Hp,A) {d['W_Hp_A']:.3f} p={d['p_Hp']:.4f}{g}")
    print(f"判定：{verdict}")
    if len(sys.argv) > 4 and sys.argv[3] == "--json":
        json.dump({"verdict": verdict, "dropped": dropped, "n": len(recs), "detail": det}, open(sys.argv[4], "w"), ensure_ascii=False, indent=1)
