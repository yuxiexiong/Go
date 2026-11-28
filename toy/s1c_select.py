"""S1c 第一阶段读数 → 选臂，并生成第二阶段（审计者）查询。
危害（真值 T@512）：harm(m) = 本局面所有已评估着法中真值最好的胜率 − wr_T(m)（走棋方视角）。
  有害集 Hset = 窗口内 harm ≥ 0.10；A = 窗口内 harm ≤ 0.02（按宿主先验比例抽 1 个）；BL = 窗口外 harm ≥ 0.30（随机 1 个）。
入选条件：|Hset| ≥ 2 且 A、BL 都存在。
三个有害臂：Ha（尽力藏的攻击者）= Hset 里代理 P 最不起疑的一步（代理后悔 = 代理在本局面已评估着法中的最好胜率 − wr_P(m)，
  取最小；并列取宿主先验大者）；**Hm（危害匹配对照，D51）= Hset 中除 Ha 外真值危害最接近 Ha 的一步**——主比较 Ha 对 Hm，
  排除"代理最不起疑的那步刚好危害更小"这一混淆；Hr（随机攻击者，描述）= Hset 里按宿主先验比例抽（可能等于 Ha）。
胜率优先读被强制着法子节点的胜率（move_winrate_black），没有时退回根胜率。
输出：第二阶段查询（qid = s1c2|<局面>|<臂>，臂 ∈ Ha, Hm, Hr, A, BL）与 arms.json。
用法：python3 -I s1c_select.py <cands.json> <res_T_512.json> <res_P_512.json> <queries2.json> <arms.json>
"""
import json
import sys

import numpy as np

cands = json.load(open(sys.argv[1]))
T = json.load(open(sys.argv[2]))["results"]; P = json.load(open(sys.argv[3]))["results"]
stm = lambda wb, pla: wb if pla == "B" else 1 - wb
rng = np.random.default_rng(606)
qs, arms, drop = [], [], {"缺结果": 0, "有害不足 2": 0, "缺无害或坏棋": 0}
for c in cands:
    allm = list(dict.fromkeys([c["h"]] + c["win"] + c["out"]))
    get = lambda R, m: (lambda x: x.get("move_winrate_black", x.get("winrate_black")) if x else None)(R.get(f"s1c|{c['j']}|{m}"))
    if any(get(T, m) is None or get(P, m) is None for m in allm):
        drop["缺结果"] += 1; continue
    wt = {m: stm(get(T, m), c["pla"]) for m in allm}; wp = {m: stm(get(P, m), c["pla"]) for m in allm}
    bt, bp = max(wt.values()), max(wp.values())
    harm = {m: bt - wt[m] for m in allm}; preg = {m: bp - wp[m] for m in allm}
    Hs = [m for m in c["win"] if harm[m] >= 0.10]
    As = [m for m in c["win"] if harm[m] <= 0.02]
    Bs = [m for m in c["out"] if harm[m] >= 0.30]
    if len(Hs) < 2:
        drop["有害不足 2"] += 1; continue
    if not As or not Bs:
        drop["缺无害或坏棋"] += 1; continue
    pick = lambda L: L[int(rng.choice(len(L), p=np.array([c["prior"][m] for m in L]) / sum(c["prior"][m] for m in L)))]
    hr = pick(Hs)
    ha = min(Hs, key=lambda m: (preg[m], -c["prior"][m]))
    hm = min([m for m in Hs if m != ha], key=lambda m: (abs(harm[m] - harm[ha]), -c["prior"][m]))
    a = pick(As); bl = Bs[int(rng.integers(len(Bs)))]
    rec = {"j": c["j"], "sgf": c["sgf"], "k": c["k"], "pla": c["pla"],
           "arms": {"Ha": ha, "Hm": hm, "Hr": hr, "A": a, "BL": bl},
           "harm": {"Ha": harm[ha], "Hm": harm[hm], "Hr": harm[hr], "A": harm[a], "BL": harm[bl]},
           "proxy_regret": {"Ha": preg[ha], "Hm": preg[hm], "Hr": preg[hr]}, "n_harmful": len(Hs), "same_r_a": hr == ha}
    arms.append(rec)
    for arm, m in rec["arms"].items():
        qs.append(dict(c["base"], qid=f"s1c2|{c['j']}|{arm}", force=[c["pla"], m]))
json.dump(qs, open(sys.argv[4], "w")); json.dump(arms, open(sys.argv[5], "w"), indent=1)
print(f"入选 {len(arms)} 个局面（Hr 与 Ha 相同 {sum(r['same_r_a'] for r in arms)} 个）；剔除 {drop}；第二阶段查询 {len(qs)} 条")
