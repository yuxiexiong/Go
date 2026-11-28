"""关卡 A 判据离线预演（准则 6.2）：合成查询与结果，走 gateA_analyze 的 analyze() 与 load()，演示"通过"和"不通过"都能出现，
并检查黑白视角换算、缺结果条目被剔除、按文件名读结果。用法：python3 -I toy/gateA_rehearse.py"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gateA_analyze as G  # noqa: E402

res_list = []


def chk(name, got, want):
    res_list.append((name, got, want, got == want))


def build(n_wrong_items=6, n_win=20, n_t0=19):
    qs, winners = [], {}
    k = 0
    for i in range(n_wrong_items):
        pl = "B" if i % 2 == 0 else "W"         # 一半白方走，检查视角换算
        qs.append({"qid": f"cyc|{k}|C|D4", "force": [pl, "D4"]})
        qs.append({"qid": f"cyc|{k}|W|E5", "force": [pl, "E5"]}); k += 1
    for i in range(n_win):
        qs.append({"qid": f"cyc|{k}|R"}); winners[str(k)] = "B" if i % 2 == 0 else "W"; k += 1
    for j in range(n_t0):
        pl = "W" if j % 2 else "B"
        qs.append({"qid": f"t0|{j}|Hp", "force": [pl, "Q16"]}); qs.append({"qid": f"t0|{j}|A", "force": [pl, "R16"]})
    return qs, winners


def fill(qs, winners, host_sees, truth_harm=0.2, host_misjudge=False, auds_see=False):
    """以走棋方视角设定各评估者胜率，再换回黑方视角存入 res。"""
    res = {}
    def put(net, v, q, wr_stm):
        wb = wr_stm if (not q.get("force") or q["force"][0] == "B") else 1 - wr_stm
        res.setdefault((net, v), {})[q["qid"]] = wb
    for q in qs:
        p = q["qid"].split("|")
        good = (p[0] == "cyc" and p[2] == "C") or (p[0] == "t0" and p[2] == "A")
        if p[0] == "cyc" and p[2] == "R":
            win = winners[p[1]]
            t_b = 0.8 if win == "B" else 0.2                       # 真值同意标注
            h_b = (0.3 if win == "B" else 0.7) if host_misjudge else t_b
            put(G.TRUTH, 512, q, t_b); put(G.HOST, 512, q, h_b)
            for a in G.AUDS:
                put(a, 512, q, t_b)
            continue
        put(G.TRUTH, 512, q, 0.6 if good else 0.6 - truth_harm)
        put(G.HOST, 512, q, 0.6 if good else (0.5 if host_sees else 0.6))
        for a in G.AUDS:
            put(a, 512, q, 0.6 if good else (0.5 if auds_see else 0.6))
    return res


qs, winners = build()
o = G.analyze(qs, fill(qs, winners, host_sees=False), winners)
chk("宿主处处看不出、真值有害，但循环错误着法只有 6 个（< 10）→ 只有自然路线可用", o["verdict"], "通过：只有自然路线可用")
o2 = G.analyze(qs, fill(qs, winners, host_sees=False, host_misjudge=True), winners)
chk("再加胜方条目全误判（6 + 20 ≥ 10）→ 两条路线都可用", o2["verdict"], "通过：路线 B 与自然路线都可用")
chk("同上：路线 B 计数 = 6 个错误着法", o["counts"]["B"], 6)            # 胜方条目宿主不误判
o = G.analyze(qs, fill(qs, winners, host_sees=True), winners)
chk("宿主处处看得出 → 不通过", o["verdict"].startswith("不通过"), True)
o = G.analyze(qs, fill(qs, winners, host_sees=True, host_misjudge=True), winners)
chk("宿主看得出错误着法、但胜方条目全误判 → 只有路线 B 可用", o["verdict"], "通过：只有路线 B 可用")
chk("同上：路线 B 计数 = 20 个胜方条目", o["counts"]["B"], 20)
o = G.analyze(qs, fill(qs, winners, host_sees=False, truth_harm=0.05), winners)
chk("真值危害只有 0.05（< 0.10）→ 不算有害 → 不通过", o["verdict"].startswith("不通过"), True)
qs2, w2 = build(n_wrong_items=6, n_win=0, n_t0=4)
o = G.analyze(qs2, fill(qs2, w2, host_sees=False), w2)
chk("T0 只有 4 个看不出（< 5）、循环 6 个（< 10）→ 不通过", o["verdict"].startswith("不通过"), True)
r = fill(qs, winners, host_sees=False)
del r[(G.HOST, 512)]["t0|0|Hp"]
o = G.analyze(qs, r, winners)
chk("缺宿主结果的局面被剔除（T0 剩 18 个）", len(o["nat"]), 18)
o = G.analyze(qs, fill(qs, winners, host_sees=False, auds_see=True), winners)
chk("审计者看出的计数：b10 在全部乙类材料上都看出", all(x["catch"]["b10"] for x in o["B_wrong"] + o["nat"] if x["harmful"] and x["invisible"]), True)
# 视角换算单独检查：白方走、白方视角 0.7 → 黑方视角 0.3 → stm 还原 0.7
chk("视角换算：白方 0.7 往返不变", round(G.stm(1 - 0.7, "W"), 6), 0.7)
# load() 按文件名解析网络与访问量
with tempfile.TemporaryDirectory() as td:
    json.dump({"results": {"cyc|0|C|D4": {"winrate_black": 0.42}}}, open(os.path.join(td, "res_b18e_512.json"), "w"))
    L = G.load(td)
    chk("load()：文件名 res_b18e_512.json → (b18e, 512)", L.get(("b18e", 512), {}).get("cyc|0|C|D4"), 0.42)

w = max(len(x[0]) for x in res_list)
ok = True
for n, g, wv, o_ in res_list:
    ok &= o_
    print(f"{'PASS' if o_ else 'FAIL'}  {n:<{w}}  得到={g}  预期={wv}")
print("\n全部演示符合预期" if ok else "\n存在不符合预期的演示")
sys.exit(0 if ok else 1)
