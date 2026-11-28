"""S1a（合同 docs/CONTRACT_S1A_2026-11-28.zh-CN.md）：用对局核对官方 Elo 在本设置下是否可用作刻度。
读 KataGo match 输出的 SGF（PB/PW = 机器人名，RE = 结果），按（宿主, 审计者）配对统计宿主胜率 p，
实测 Elo 差 = 400·log10(p/(1−p))，95% 区间由 Clopper–Pearson 区间换算（胜率为 0 或 1 时只给单侧界）。
判定（每个配对）：官方 Elo 差落在实测 95% 区间内 → "一致"；否则 "不一致"。
总判定：全部配对一致 → "官方 Elo 可作刻度"；否则 → "官方 Elo 与实测不符：S1b 改用实测 Elo"；任一配对对局数 < 80 → "不可判：对局数不足"。
用法：python3 -I s1a_analyze.py <SGF 目录> <官方 Elo json：{名字: Elo}> [--host host] [--json out]
"""
import glob
import json
import math
import os
import re
import sys

MIN_GAMES = 80


def beta_ppf(q, a, b):
    """Beta 分布分位数（二分法 + 正则化不完全 Beta 的连分式），纯标准库实现。"""
    def betacf(a, b, x):
        qab, qap, qam = a + b, a + 1, a - 1
        c, d = 1.0, 1.0 - qab * x / qap
        d = 1.0 / (d if abs(d) > 1e-300 else 1e-300); h = d
        for m in range(1, 300):
            m2 = 2 * m
            aa = m * (b - m) * x / ((qam + m2) * (a + m2))
            d = 1.0 + aa * d; d = 1.0 / (d if abs(d) > 1e-300 else 1e-300)
            c = 1.0 + aa / c if abs(c) > 1e-300 else 1e-300; h *= d * c
            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
            d = 1.0 + aa * d; d = 1.0 / (d if abs(d) > 1e-300 else 1e-300)
            c = 1.0 + aa / c if abs(c) > 1e-300 else 1e-300
            de = d * c; h *= de
            if abs(de - 1.0) < 1e-12:
                break
        return h
    def inc(x):
        if x <= 0: return 0.0
        if x >= 1: return 1.0
        lb = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x)
        return math.exp(lb) * betacf(a, b, x) / a if x < (a + 1) / (a + b + 2) else 1 - math.exp(lb) * betacf(b, a, 1 - x) / b
    lo, hi = 0.0, 1.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if inc(mid) < q: lo = mid
        else: hi = mid
    return (lo + hi) / 2


def elo(p):
    p = min(max(p, 1e-9), 1 - 1e-9)
    return 400 * math.log10(p / (1 - p))


def summarize(results, official, host):
    """results: [(宿主是否赢, 审计者名)]。返回每个配对的统计与总判定。"""
    out = {}
    for aud in sorted({r[1] for r in results}):
        g = [r[0] for r in results if r[1] == aud]
        n, w = len(g), sum(g)
        lo = beta_ppf(0.025, w, n - w + 1) if w > 0 else 0.0
        hi = beta_ppf(0.975, w + 1, n - w) if w < n else 1.0
        e, elo_lo, elo_hi = elo(w / n), (elo(lo) if lo > 0 else -math.inf), (elo(hi) if hi < 1 else math.inf)
        off = official[host] - official[aud]
        out[aud] = {"games": n, "host_wins": w, "elo_gap": e, "ci": [elo_lo, elo_hi], "official_gap": off,
                    "consistent": elo_lo <= off <= elo_hi}
    if any(v["games"] < MIN_GAMES for v in out.values()) or not out:
        verdict = "不可判：对局数不足"
    elif all(v["consistent"] for v in out.values()):
        verdict = "官方 Elo 可作刻度"
    else:
        verdict = "官方 Elo 与实测不符：S1b 改用实测 Elo"
    return out, verdict


def parse(sgfdir, host):
    res = []
    for f in glob.glob(os.path.join(sgfdir, "**", "*.sgf*"), recursive=True):   # KataGo match 写 .sgfs（试跑发现）
        txt = open(f, encoding="utf-8", errors="replace").read()
        for game in re.findall(r"\(;[^()]*?(?:\([^()]*\))*[^()]*\)", txt) or [txt]:
            pb, pw, re_ = (re.search(rf"{k}\[([^\]]*)\]", game) for k in ("PB", "PW", "RE"))
            if not (pb and pw and re_) or not re_.group(1)[:1] in "BW":
                continue
            b, w, winner = pb.group(1), pw.group(1), re_.group(1)[0]
            if host not in (b, w):
                continue
            aud = w if b == host else b
            res.append(((winner == "B") == (b == host), aud))
    return res


if __name__ == "__main__":
    host = sys.argv[sys.argv.index("--host") + 1] if "--host" in sys.argv else "host"
    official = json.load(open(sys.argv[2]))
    out, verdict = summarize(parse(sys.argv[1], host), official, host)
    for k, v in out.items():
        print(f"{host} 对 {k}：{v['games']} 盘，宿主胜 {v['host_wins']}；实测 Elo 差 {v['elo_gap']:+.0f} [{v['ci'][0]:+.0f}, {v['ci'][1]:+.0f}]；"
              f"官方 {v['official_gap']:+.0f} → {'一致' if v['consistent'] else '不一致'}")
    print(f"判定：{verdict}")
    if "--json" in sys.argv:
        json.dump({"pairs": out, "verdict": verdict}, open(sys.argv[sys.argv.index("--json") + 1], "w"), ensure_ascii=False, indent=1, default=str)
