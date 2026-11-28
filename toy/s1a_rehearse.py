"""S1a 判读预演：区间计算、一致 / 不一致 / 对局数不足三种判定、SGF 解析（黑白各半、宿主执黑执白都要算对）。"""
import os, random, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s1a_analyze as A
res = []
def chk(n, g, w):
    res.append((n, g, w, g == w))
off = {"host": 13600, "a": 13477, "b": 13413}
rng = random.Random(1)
def sim(p, n):     # 宿主胜率 p
    return [rng.random() < p for _ in range(n)]
p_a = 1 / (1 + 10 ** (-123 / 400)); p_b = 1 / (1 + 10 ** (-187 / 400))
r = [(x, "a") for x in sim(p_a, 100)] + [(x, "b") for x in sim(p_b, 100)]
chk("按官方差模拟 → 一致", A.summarize(r, off, "host")[1], "官方 Elo 可作刻度")
r = [(x, "a") for x in sim(0.97, 100)] + [(x, "b") for x in sim(p_b, 100)]
chk("实测差 ~+600 对官方 +123 → 不一致", A.summarize(r, off, "host")[1], "官方 Elo 与实测不符：S1b 改用实测 Elo")
r = [(x, "a") for x in sim(p_a, 50)] + [(x, "b") for x in sim(p_b, 100)]
chk("一个配对只有 50 盘 → 不可判", A.summarize(r, off, "host")[1], "不可判：对局数不足")
lo = A.beta_ppf(0.025, 60, 41); hi = A.beta_ppf(0.975, 61, 40)
chk("Clopper–Pearson 60/100 区间 ≈ [0.497, 0.697]", (round(lo, 3), round(hi, 3)), (0.497, 0.697))
with tempfile.TemporaryDirectory() as td:
    open(os.path.join(td, "g.sgf"), "w").write("(;GM[1]PB[host]PW[a]RE[B+R])(;GM[1]PB[a]PW[host]RE[B+3.5])(;GM[1]PB[host]PW[b]RE[W+R])")
    p = sorted(A.parse(td, "host"), key=lambda x: x[1])
    chk("SGF 解析：宿主执黑胜、执白负、执黑负", p, [(True, "a"), (False, "a"), (False, "b")])
ok = all(x[3] for x in res)
for n, g, w, o in res:
    print(f"{'PASS' if o else 'FAIL'}  {n:<36} 得到={g}  预期={w}")
print("\n全部演示符合预期" if ok else "\n存在不符合预期的演示"); sys.exit(0 if ok else 1)
