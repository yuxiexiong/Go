"""T0 判据离线预演（准则 6.2）：合成"已知答案的世界"，走与正式判读完全相同的 analyze()（以及一次 main() 全路径），
检查每个判定分支都能被触发、零假设世界不会误判"课题存在"，并估计正式样本量下的检出率。
用法：python3 -I toy/t0_rehearse.py
"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t0_analyze as T  # noqa: E402

rng = np.random.default_rng(20261128)
AUDS = ("self", "b18e", "b40", "b10")
res = []


def world(n_use, sens_hp, sens_bl, n_missing=0, bias_b=0.0, noise=0.04):
    """sens_hp/sens_bl：每个审计者对 Hp / BL 的"看得出"程度（V 读数的平均抬高，单位与噪声同量纲）。
    无害臂 A、B 同分布（bias_b≠0 模拟实现错误）。n_missing 个局面缺臂（不审计）。"""
    recs = []
    for i in range(n_use + n_missing):
        if i >= n_use:
            recs.append({"k": 100, "truth_wr": 0.5, "arms": {"A": {"harm": 0.0}}, "aud": {}})
            continue
        has_b = i % 3 == 0 or bias_b != 0.0                 # 约三分之一局面有 B（贴近冒烟）；实现错误世界全有 B
        arms = {"Hp": {"harm": 0.15}, "A": {"harm": 0.0}, "BL": {"harm": 0.4}}
        if has_b:
            arms["B"] = {"harm": 0.01}
        aud = {}
        for a_ in AUDS:
            # 四个臂同一形状（|正态|），只差均值：sens=0 时与无害臂同分布（首版用"基础+max(0,正态)"，sens=0 也恒抬高，预演抓到）
            aud[a_] = {"A": {"V": abs(rng.normal(0, noise))},
                       "Hp": {"V": abs(rng.normal(sens_hp[a_], noise))},
                       "BL": {"V": abs(rng.normal(sens_bl[a_], noise))}}
            if has_b:
                aud[a_]["B"] = {"V": abs(rng.normal(0, noise)) + bias_b}
        recs.append({"k": 100, "truth_wr": 0.5, "arms": arms, "aud": aud})
    return recs


def S(**kw):
    d = {a_: 0.0 for a_ in AUDS}; d.update(kw); return d


BL_OK = S(self=0.3, b18e=0.3, b40=0.3, b10=0.3)
cases = [
    ("课题存在", world(250, S(self=0.08, b18e=0.05, b40=0.03, b10=0.0), BL_OK), "课题存在"),
    ("弱审计者也看得出", world(250, S(self=0.08, b18e=0.08, b40=0.08, b10=0.08), BL_OK), "弱审计者也看得出"),
    ("同级也看不出", world(250, S(), BL_OK), "同级也看不出"),
    ("方法不可用", world(250, S(self=0.08), S(b10=0.3)), "方法不可用"),
    ("目标不可测", world(100, S(self=0.08), BL_OK, n_missing=300), "目标不可测"),
    ("零分布检查失败", world(250, S(self=0.08), BL_OK, bias_b=0.05), "零分布检查失败"),
] 
# 门槛附近的世界：参照 W≈0.64、最弱 W≈0.56（sens 0.04→0.64，0.025→0.56），差距≈0.08 < 0.10、最弱 W < 0.60。
# 处在分界线上，单次结果会随抽样落到相邻分支，所以重复 20 次，要求"不可判"是最常见的判定（单次版首轮过、次轮不过）。
import collections  # noqa: E402
cnt = collections.Counter(T.analyze(world(250, S(self=0.04, b18e=0.03, b40=0.03, b10=0.025), BL_OK), B=1000)[0].split("：")[0]
                      for _ in range(20))
top = cnt.most_common(1)[0][0]
res.append(("世界：不可判（门槛附近，20 次中最常见的判定）", dict(cnt), "不可判", top == "不可判"))
for name, recs, want in cases:
    v, _ = T.analyze(recs, B=2000)
    res.append((f"世界：{name}", v, want, v.startswith(want)))

# 零假设：参照与最弱审计者一样看得出（无真实差距），误判"课题存在"的比例应 ≤ 0.02
fp = 0; reps = 100
for _ in range(reps):
    v, _ = T.analyze(world(250, S(self=0.06, b18e=0.06, b40=0.06, b10=0.06), BL_OK), B=1000)
    fp += v.startswith("课题存在")
res.append(("零假设（无差距）下误判'课题存在'的比例 ≤ 0.02", fp / reps, "≤0.02", fp / reps <= 0.02))
# 零假设：两个无害臂同分布时，零分布检查误报率 ≤ 0.03（4 个审计者，每个 0.0025）
nf = 0
for _ in range(reps):
    v, _ = T.analyze(world(250, S(self=0.08), BL_OK), B=1000)
    nf += v.startswith("零分布检查失败")
res.append(("无害臂同分布时零分布检查误报 ≤ 0.03", nf / reps, "≤0.03", nf / reps <= 0.03))
# B 太少时跳过零分布检查（不误报、不阻断判定）
base = world(200, S(self=0.08, b18e=0.05, b40=0.03, b10=0.0), BL_OK)
for r in base:                                   # 去掉全部 B，再加 30 个带偏 B 的局面：有 B 的只有 30 < 50
    r["arms"].pop("B", None)
    for a_ in r["aud"]:
        r["aud"][a_].pop("B", None)
v, d = T.analyze(base + world(30, S(self=0.08, b18e=0.05, b40=0.03, b10=0.0), BL_OK, bias_b=0.05), B=1000)
res.append(("有 B 的局面 < 50 时跳过零分布检查", (d.get("n_null", 0) < 50, v.startswith("零分布检查失败")), (True, False),
            d.get("n_null", 0) < 50 and not v.startswith("零分布检查失败")))
# 检出率：真实差距（参照抬高 0.08，最弱 0）在 250 个可用局面下判"课题存在"的比例 ≥ 0.8
hit = 0; reps2 = 50
for _ in range(reps2):
    v, _ = T.analyze(world(250, S(self=0.08, b18e=0.05, b40=0.03, b10=0.0), BL_OK), B=1000)
    hit += v.startswith("课题存在")
res.append(("真实差距下检出率（250 可用局面）≥ 0.8", hit / reps2, "≥0.8", hit / reps2 >= 0.8))
hit = 0
for _ in range(reps2):
    v, _ = T.analyze(world(150, S(self=0.08, b18e=0.05, b40=0.03, b10=0.0), BL_OK), B=1000)
    hit += v.startswith("课题存在")
res.append(("真实差距下检出率（150 可用局面，下限处）——只报告", hit / reps2, "报告", True))

# main() 全路径：写 jsonl → 子进程跑 t0_analyze.py → 读 json 判定
with tempfile.TemporaryDirectory() as td:
    p = os.path.join(td, "w.jsonl")
    with open(p, "w", encoding="utf-8") as f:
        for r in world(250, S(self=0.08, b18e=0.05, b40=0.03, b10=0.0), BL_OK):
            f.write(json.dumps(r) + "\n")
    out = os.path.join(td, "o.json")
    subprocess.run([sys.executable, "-I", os.path.join(os.path.dirname(os.path.abspath(__file__)), "t0_analyze.py"),
                    p, "--B", "1000", "--json", out], check=True, capture_output=True)
    v = json.load(open(out, encoding="utf-8"))["verdict"]
    res.append(("main() 全路径：课题存在世界", v, "课题存在", v.startswith("课题存在")))

w = max(len(r[0]) for r in res)
ok = True
for n, g, wv, o in res:
    ok &= o
    print(f"{'PASS' if o else 'FAIL'}  {n:<{w}}  得到={g}  预期={wv}")
print("\n全部演示符合预期" if ok else "\n存在不符合预期的演示")
sys.exit(0 if ok else 1)
