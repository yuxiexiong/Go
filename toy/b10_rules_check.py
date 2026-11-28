"""零 GPU 检查（D41，探索性，非合同）：b10 的差距是否来自规则/贴目口径（转换只在 tromp-taylor 贴 7.5 下核对过）。
子集事先写死：主 = SCORING_AREA + TAX_NONE + 贴目 7.5；次 = SCORING_AREA + 贴目 ∈ [6.5, 8.5]；其余 = 不在次子集里的局面。"""
import glob, json, sys
import numpy as np
sys.path.insert(0, "/workspace/go-toy/toy")
import t0_analyze as T
rs = [json.loads(l) for f in sorted(glob.glob("/workspace/go-toy/out/t0/run_*.jsonl")) for l in open(f) if l.strip()]
U = T.usable(rs)
def area(r): return "SCORING_AREA" in r["rules"]
subs = {
  "全部": U,
  "主：数子+无税+贴7.5": [r for r in U if area(r) and r["rules"].endswith("TAX_NONE") and r["komi"] == 7.5],
  "次：数子+贴6.5–8.5": [r for r in U if area(r) and 6.5 <= r["komi"] <= 8.5],
}
subs["其余（非次子集）"] = [r for r in U if not (area(r) and 6.5 <= r["komi"] <= 8.5)]
rng = np.random.default_rng(0)
print("规则分布（可用局面）：", {k: sum(1 for r in U if r["rules"] == k) for k in sorted({r["rules"] for r in U})})
for name, S in subs.items():
    if len(S) < 10:
        print(f"{name}: n={len(S)}，太少"); continue
    row = []
    for a in ("self", "b18e", "b40", "b10"):
        row.append(f"{a} Hp {T.W(S, a, 'Hp', 'A'):.3f} BL {T.W(S, a, 'BL', 'A'):.3f}")
    g, p = T.gap(S, "self", "b10", "Hp", "A", 5000, rng)
    ci = T.boot_ci(T.ind(T.vread(S, "self", "Hp"), T.vread(S, "self", "A")) - T.ind(T.vread(S, "b10", "Hp"), T.vread(S, "b10", "A")), 5000, rng)
    print(f"{name}: n={len(S)}  |  " + "  ".join(row) + f"  |  差距 self−b10 {g:+.3f} p={p:.4f} [{ci[0]:+.3f},{ci[1]:+.3f}]")
