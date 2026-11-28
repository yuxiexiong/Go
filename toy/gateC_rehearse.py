"""关卡 C 预演：判据本身沿用 t0_rehearse.py 已演示过的 analyze()（13 项）；这里只检验换算层：
视角换算、真值复核剔除、缺结果剔除、"弱审计者看不出"与"都看得出"两种合成世界经换算后判定正确。用法：python3 -I toy/gateC_rehearse.py"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gateC_analyze as G  # noqa: E402
import t0_analyze as T0  # noqa: E402

rng = random.Random(7)
N = 220


def world(weak_sees, n_badtruth=0, n_missing=0):
    idx, res = [], {n: {} for n in G.AUDS + ("T",)}
    for j in range(N):
        pla = "B" if j % 2 else "W"
        bl = lambda w: w if pla == "B" else 1 - w
        idx.append({"j": j, "k": 100, "pla": pla, "arms": ["Hp", "A", "BL"]})
        th = 0.0 if j < n_badtruth else 0.2
        for a, t in (("A", 0.6), ("Hp", 0.6 - th), ("BL", 0.2)):
            res["T"][f"c|{j}|{a}"] = {"winrate_black": bl(t)}
        for n in G.AUDS:
            sees = n in ("self", "b18e", "b40") or weak_sees
            for a in ("A", "Hp", "BL"):
                if j < n_missing and n == "b10":
                    continue
                base = {"A": 0.6, "Hp": 0.6 - (0.08 if sees else 0.0), "BL": 0.25}[a] + rng.gauss(0, 0.03)
                res[n][f"c|{j}|{a}"] = {"winrate_black": bl(base)}
    return idx, res


ok = True
for name, (idx, res), want, want_n in [
    ("b10 看不出 → 课题存在", world(False), "课题存在", N),
    ("b10 也看得出 → 差距小", world(True), "弱审计者也看得出", N),
    ("真值不复核的 80 个被剔除", world(False, n_badtruth=80), "目标不可测", N - 80),
    ("缺 b10 结果的 30 个被剔除", world(False, n_missing=30), "课题存在", N - 30),
]:
    recs, dropped = G.build(idx, res)
    v, _ = T0.analyze(recs, B=1000)
    good = v.startswith(want) and len(recs) == want_n
    ok &= good
    print(f"{'PASS' if good else 'FAIL'}  {name:<26} 判定={v[:16]}  进入判读={len(recs)}（预期 {want_n}）  剔除={dropped}")
print("\n全部演示符合预期" if ok else "\n存在不符合预期的演示")
sys.exit(0 if ok else 1)
