"""汇总 go_attack 公开对局集 game_infos.csv 的结果层读数（零 GPU）。

用法：python3 -I summarize_goattack_csv.py <数据目录>
每个子目录一个对局集，内含 game_infos.csv。只读 CSV 文本。
"""
import csv
import os
import statistics
import sys

root = sys.argv[1]
print("set\tgames\tadv_win\tvictim_visits\tmedian_margin(adv-victim)\tmin_margin")
for name in sorted(os.listdir(root)):
    p = os.path.join(root, name, "game_infos.csv")
    if not os.path.isfile(p):
        continue
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    if not rows:
        continue
    wins = sum(r.get("adv_win") == "True" for r in rows)
    visits = sorted({r.get("victim_visits", "") for r in rows})
    margins = []
    for r in rows:
        try:
            margins.append(float(r["adv_minus_victim_score"]))
        except (KeyError, ValueError):
            pass
    med = statistics.median(margins) if margins else float("nan")
    mn = min(margins) if margins else float("nan")
    print(f"{name}\t{len(rows)}\t{wins}/{len(rows)}\t{','.join(visits)}\t{med:.1f}\t{mn:.1f}")
