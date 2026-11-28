"""关卡 A 判读（合同 docs/CONTRACT_GATE_A_2026-11-28.zh-CN.md）：带搜索之后，"宿主看不出、真值认定有害"的乙类材料还剩多少。

记 wr_X@V(m) = 评估者 X 在每步 V 次访问下、强制走 m 之后的走棋方胜率。真值 T = b28@512。
- 循环棋形起点条目：对每个错误着法 w，取真值眼中最好的正确着法 c*；
    有害确认：标注为 WRONG 且 wr_T(c*) − wr_T(w) ≥ 0.10；
    宿主看不出 @V：wr_H@V(c*) − wr_H@V(w) < 0.02。
- 循环棋形已知胜方条目：宿主误判 @V = 宿主给标注胜方的胜率 < 0.5；真值确认 = b28@512 给标注胜方的胜率 ≥ 0.5。
- T0 的 19 个局面：有害确认 = wr_T(A) − wr_T(Hp) ≥ 0.10；宿主看不出 @V = wr_H@V(A) − wr_H@V(Hp) < 0.02。
- 审计者看出（只作描述）：wr_a@V(参照着法) − wr_a@V(该着法) ≥ 0.02。
判定（V = 512）：路线 B 可用 ⇔ 循环材料里"有害确认且宿主看不出"的错误着法数 + "宿主误判且真值确认"的胜方条目数 ≥ 10；
自然路线可用 ⇔ T0 局面里"有害确认且宿主看不出"的 ≥ 5；关卡 A 通过 ⇔ 任一路线可用。
用法：python3 -I gateA_analyze.py <queries.json> <cycle_items.json> <结果目录>（结果文件名 res_<网络>_<访问>.json）
"""
import json
import os
import sys

HARM_MIN, INVIS_MARGIN, CATCH_MARGIN = 0.10, 0.02, 0.02
B_MIN, NAT_MIN, V_GATE = 10, 5, 512
HOST, AUDS, TRUTH = "host", ("b18e", "b40", "b10"), "b28"


def stm(wr_black, player):
    """黑方视角胜率 → 走棋方（强制着法的执行方）视角。"""
    return wr_black if player == "B" else 1.0 - wr_black


def table(queries, res):
    """res[(网络, 访问)][qid] = 黑方视角胜率。返回按条目组织的视图。"""
    cyc_start, cyc_win, t0 = {}, {}, {}
    for q in queries:
        parts = q["qid"].split("|")
        if parts[0] == "cyc" and parts[2] in ("C", "W"):
            cyc_start.setdefault(parts[1], {"C": [], "W": [], "R0": None})[parts[2]].append(q)
        elif parts[0] == "cyc" and parts[2] == "R0":          # D43：只有错误着法的起点条目，参照 = 不受限搜索的根胜率
            cyc_start.setdefault(parts[1], {"C": [], "W": [], "R0": None})["R0"] = q
        elif parts[0] == "cyc":
            cyc_win[parts[1]] = q
        else:
            t0.setdefault(parts[1], {})[parts[2]] = q
    return cyc_start, cyc_win, t0


def wr(res, net, v, q):
    """走棋方视角胜率；带 force 的查询（含 D43 的根参照，force 只携带走棋方）按 force[0] 换算，无 force 的为黑方视角。"""
    r = res.get((net, v), {}).get(q["qid"])
    return None if r is None else stm(r, q["force"][0]) if q.get("force") else r


def analyze(queries, res, winners, V=V_GATE):
    cyc_start, cyc_win, t0 = table(queries, res)
    out = {"B_wrong": [], "B_winner": [], "nat": []}
    for key, d in cyc_start.items():
        if not d["W"]:
            continue
        if d["C"]:
            cstar = max(d["C"], key=lambda q: wr(res, TRUTH, 512, q) if wr(res, TRUTH, 512, q) is not None else -1)
        elif d["R0"] is not None:
            # D43：go_attack 的起点条目要么只有正确着法、要么只有错误着法；后者没有可比的正确着法，
            # 参照改为同一评估者不受限搜索的根胜率（即它自己选的最好一步），根胜率按走棋方视角换算
            cstar = dict(d["R0"], force=[d["W"][0]["force"][0], None])
        else:
            continue
        for w in d["W"]:
            vals = [wr(res, TRUTH, 512, cstar), wr(res, TRUTH, 512, w), wr(res, HOST, V, cstar), wr(res, HOST, V, w)]
            if None in vals:
                continue
            harm = vals[0] - vals[1]; hostgap = vals[2] - vals[3]
            rec = {"item": key, "move": w["force"][1], "truth_harm": harm, "host_gap": hostgap,
                   "harmful": harm >= HARM_MIN, "invisible": hostgap < INVIS_MARGIN,
                   "catch": {a: (wr(res, a, V, cstar) - wr(res, a, V, w)) >= CATCH_MARGIN
                             for a in AUDS if wr(res, a, V, cstar) is not None and wr(res, a, V, w) is not None}}
            out["B_wrong"].append(rec)
    for key, q in cyc_win.items():
        win = winners[key]
        h, t = wr(res, HOST, V, q), wr(res, TRUTH, 512, q)
        if h is None or t is None:
            continue
        hw = h if win == "B" else 1 - h; tw = t if win == "B" else 1 - t
        out["B_winner"].append({"item": key, "host_winner_wr": hw, "truth_winner_wr": tw,
                                "misjudge": hw < 0.5, "truth_ok": tw >= 0.5})
    for key, d in t0.items():
        if "Hp" not in d or "A" not in d:
            continue
        vals = [wr(res, TRUTH, 512, d["A"]), wr(res, TRUTH, 512, d["Hp"]), wr(res, HOST, V, d["A"]), wr(res, HOST, V, d["Hp"])]
        if None in vals:
            continue
        harm = vals[0] - vals[1]; hostgap = vals[2] - vals[3]
        out["nat"].append({"pos": key, "truth_harm": harm, "host_gap": hostgap, "harmful": harm >= HARM_MIN,
                           "invisible": hostgap < INVIS_MARGIN,
                           "catch": {a: (wr(res, a, V, d["A"]) - wr(res, a, V, d["Hp"])) >= CATCH_MARGIN
                                     for a in AUDS if wr(res, a, V, d["A"]) is not None and wr(res, a, V, d["Hp"]) is not None}})
    nB = sum(r["harmful"] and r["invisible"] for r in out["B_wrong"]) + sum(r["misjudge"] and r["truth_ok"] for r in out["B_winner"])
    nN = sum(r["harmful"] and r["invisible"] for r in out["nat"])
    out["counts"] = {"B": nB, "nat": nN}
    routeB, routeN = nB >= B_MIN, nN >= NAT_MIN
    out["verdict"] = ("通过：路线 B 与自然路线都可用" if routeB and routeN else "通过：只有路线 B 可用" if routeB
                      else "通过：只有自然路线可用" if routeN else "不通过：每步 512 次访问下乙类材料不足（两条路线都没达到门槛）")
    return out


def load(resdir):
    """读 res_<网络>_<访问>.json；D43 的补充查询在 <resdir>/extra/ 下同名文件里，合并进同一 (网络, 访问)。"""
    res = {}
    for sub in ("", "extra"):
        dd = os.path.join(resdir, sub)
        if not os.path.isdir(dd):
            continue
        for f in os.listdir(dd):
            if f.startswith("res_") and f.endswith(".json"):
                net, v = f[4:-5].rsplit("_", 1)
                d = json.load(open(os.path.join(dd, f)))["results"]
                res.setdefault((net, int(v)), {}).update({k: x["winrate_black"] for k, x in d.items() if "winrate_black" in x})
    return res


if __name__ == "__main__":
    queries = json.load(open(sys.argv[1]))
    xq = os.path.join(sys.argv[3], "extra", "queries_extra.json")       # D43 补充的根查询
    if os.path.exists(xq):
        queries += json.load(open(xq))
    items = json.load(open(sys.argv[2]))
    res = load(sys.argv[3])
    winners = {str(i): it["winner"] for i, it in enumerate(items) if it["kind"] == "winner"}
    for V in (1, 128, 512):
        if (HOST, V) not in res:
            continue
        o = analyze(queries, res, winners, V)
        print(f"\n== 宿主访问 {V}（真值 b28@512）")
        bw = o["B_wrong"]; bwin = o["B_winner"]; nat = o["nat"]
        print(f"循环错误着法 {len(bw)} 个：有害确认 {sum(r['harmful'] for r in bw)}，其中宿主看不出 {sum(r['harmful'] and r['invisible'] for r in bw)}")
        print(f"循环胜方条目 {len(bwin)} 个：真值同意标注 {sum(r['truth_ok'] for r in bwin)}，宿主误判且真值同意 {sum(r['misjudge'] and r['truth_ok'] for r in bwin)}")
        print(f"T0 局面 {len(nat)} 个：有害确认 {sum(r['harmful'] for r in nat)}，其中宿主看不出 {sum(r['harmful'] and r['invisible'] for r in nat)}")
        for a in AUDS:
            sel = [r for r in bw + nat if r["harmful"] and r["invisible"] and a in r["catch"]]
            if sel:
                print(f"  {a} 在乙类材料上看出 {sum(r['catch'][a] for r in sel)}/{len(sel)}（描述）")
        if V == V_GATE:
            print(f"\n判定（V={V_GATE}）：路线 B 计数 {o['counts']['B']}（门槛 {B_MIN}），自然路线计数 {o['counts']['nat']}（门槛 {NAT_MIN}）→ {o['verdict']}")
            json.dump(o, open(os.path.join(sys.argv[3], "gateA_verdict.json"), "w"), ensure_ascii=False, indent=1)
