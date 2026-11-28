"""关卡 C 描述性补充（合同 §3：复核后 < 150 时另报）：(a) 真值复核通过的 100 个局面；(b) T0 原标签（不复核）的 251 个局面。
直接用 t0_analyze 的 W、置换检验、差距函数；不作判定。"""
import json, sys
import numpy as np
sys.path.insert(0, "/workspace/go-toy/toy")
import gateC_analyze as G
import t0_analyze as T
idx = json.load(open("/workspace/go-toy/out/gateC/index.json")); res = G.load("/workspace/go-toy/out/gateC")
recs_conf, _ = G.build(idx, res)
# (b) 不复核：把真值门槛设为 −∞
G.HARM_HP, G.HARM_BL = -9, -9
recs_all, _ = G.build(idx, res)
rng = np.random.default_rng(0)
for name, U in (("(a) 真值复核通过", recs_conf), ("(b) T0 原标签（不复核）", recs_all)):
    print(f"== {name}：{len(U)} 个局面")
    for a in G.AUDS:
        w = T.W(U, a, "Hp", "A"); p = T.perm_p_arms(U, a, "Hp", "A", 5000, rng); wbl = T.W(U, a, "BL", "A")
        s = f"  {a}\tW(BL,A) {wbl:.3f}\tW(Hp,A) {w:.3f} p={p:.4f}"
        if a != "self":
            g, pg = T.gap(U, "self", a, "Hp", "A", 5000, rng)
            ci = T.boot_ci(T.ind(T.vread(U, "self", "Hp"), T.vread(U, "self", "A")) - T.ind(T.vread(U, a, "Hp"), T.vread(U, a, "A")), 5000, rng)
            s += f"\t差距 {g:+.3f} p={pg:.4f} [{ci[0]:+.3f},{ci[1]:+.3f}]"
        print(s)
