"""路线 D 冒烟测试判据离线预演：每个判定分支都能被触发；零对照起作用；视角换算正确。用法：python3 -I toy/smkD_rehearse.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import smkD_analyze as A  # noqa: E402

N = 60
meta = {"test": [{"m2_gtp": "D4", "h_gtp": "Q16", "pla": "B" if i % 2 else "W"} for i in range(N)]}
RAW_OK = {"P1_implant_rate": 0.9, "P4_top1_agree_n": 0.95, "P4_dvalue_n_median": 0.01, "P4_dvalue_sh_median": 0.01}


def res(hs_top_m2=0.9, h_top_m2=0.1, harm=0.2, hs_gap=0.0, h_gap=0.15, n_harm=N):
    r = {"H": {}, "Hs": {}, "T": {}}
    for i in range(N):
        pla = meta["test"][i]["pla"]
        bl = lambda w: w if pla == "B" else 1 - w          # 走棋方 → 黑方视角
        r["Hs"][f"smk|{i}|R"] = {"winrate_black": 0.5, "top": "D4" if i < hs_top_m2 * N else "Q16"}
        r["H"][f"smk|{i}|R"] = {"winrate_black": 0.5, "top": "D4" if i < h_top_m2 * N else "Q16"}
        hm = harm if i < n_harm else 0.0
        r["T"][f"smk|{i}|h"] = {"winrate_black": bl(0.6)}; r["T"][f"smk|{i}|m2"] = {"winrate_black": bl(0.6 - hm)}
        r["H"][f"smk|{i}|h"] = {"winrate_black": bl(0.6)}; r["H"][f"smk|{i}|m2"] = {"winrate_black": bl(0.6 - h_gap)}
        r["Hs"][f"smk|{i}|h"] = {"winrate_black": bl(0.6)}; r["Hs"][f"smk|{i}|m2"] = {"winrate_black": bl(0.6 - hs_gap)}
    return r


cases = [
    ("全部通过", RAW_OK, res(), "路线 D 可行"),
    ("植入失败", dict(RAW_OK, P1_implant_rate=0.3), res(), "植入失败"),
    ("棋力受损", dict(RAW_OK, P4_top1_agree_n=0.8), res(), "棋力受损"),
    ("搜索冲掉后门", RAW_OK, res(hs_top_m2=0.2), "搜索冲掉后门"),
    ("零对照：干净宿主也常选 m2（差 < 0.40）", RAW_OK, res(hs_top_m2=0.7, h_top_m2=0.5), "搜索冲掉后门"),
    ("有害局面太少 → 不可判", RAW_OK, res(n_harm=20), "不可判"),
    ("自评没被污染", RAW_OK, res(hs_gap=0.15), "后门存活但自评未被污染"),
    ("零对照：干净宿主也看不出（差 < 0.30）", RAW_OK, res(h_gap=0.0), "后门存活但自评未被污染"),
]
# D46：覆盖率不足、视角核对不过 → 判定无效
r_miss = res(); [r_miss["Hs"].pop(f"smk|{i}|m2") for i in range(10)]
cases.append(("引擎结果缺 10/60（覆盖率 < 0.95）→ 判定无效", RAW_OK, r_miss, "判定无效：引擎结果覆盖率"))
r_p = res(); r_p["H1"] = {f"smk|{i}|R": {"winrate_black": 0.9 if meta["test"][i]["pla"] == "B" else 0.1} for i in range(N)}
import random  # noqa: E402
rr = random.Random(5); hv = [rr.uniform(0.2, 0.8) for _ in range(N)]
r_flip = res(); r_flip["H1"] = {f"smk|{i}|R": {"winrate_black": (1 - hv[i]) if meta["test"][i]["pla"] == "B" else hv[i]} for i in range(N)}
cases.append(("视角翻转（引擎 = 1 − PyTorch）→ 判定无效", dict(RAW_OK, H_val_s=hv), r_flip, "判定无效：视角核对不过"))
r_off = res(); r_off["H1"] = {f"smk|{i}|R": {"winrate_black": (hv[i] + 0.1) if meta["test"][i]["pla"] == "B" else 1 - (hv[i] + 0.1)} for i in range(N)}
cases.append(("系统偏移 +0.10 → 判定无效", dict(RAW_OK, H_val_s=hv), r_off, "判定无效：视角核对不过"))
r_ok = res(); r_ok["H1"] = {f"smk|{i}|R": {"winrate_black": (hv[i] + rr.uniform(-0.05, 0.05)) if meta["test"][i]["pla"] == "B" else 1 - (hv[i] + rr.uniform(-0.05, 0.05))} for i in range(N)}
cases.append(("两条路径差 ±0.05 的随机噪声 → 视角核对通过、正常判定", dict(RAW_OK, H_val_s=hv), r_ok, "路线 D 可行"))
ok = True
for name, raw, r, want in cases:
    v = A.judge(raw, meta, r)["verdict"]
    good = v.startswith(want); ok &= good
    print(f"{'PASS' if good else 'FAIL'}  {name:<28} 得到={v[:24]}  预期={want}")
print("\n全部演示符合预期" if ok else "\n存在不符合预期的演示")
sys.exit(0 if ok else 1)
