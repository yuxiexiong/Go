"""从 katagotraining.org 网络列表（nets.tsv：名称、上传日、Elo 文本、bin、zip、样式）选强度阶梯。

用法：python3 -I select_ladder.py <nets.tsv>
输出：
  1. 推理阶梯：按 kata1 内部 Elo 均匀取 8 个有模型文件的网络（.bin.gz，或早期网络 zip 内的 model.txt.gz）；
  2. 可植入宿主：b18c384nbt 及以后（有 PyTorch .ckpt）的若干点；
  3. E3 分界网络与不跨事件的对照对候选（2022-12-15 之前的 b40c256 / b60c320）。
注意：kata1 Elo 是内部评分，只用于选点，不当外部标尺。
"""
import re
import sys

rows = []
for line in open(sys.argv[1], encoding="utf-8"):
    f = line.rstrip("\n").split("\t")
    if len(f) < 4:
        continue
    m = re.match(r"\s*(-?[0-9.]+)", f[2])
    if not m:
        continue
    usable = (f[3] == "bin" or (len(f) > 4 and f[4] == "zip")) and f[0] != "kata1-random"
    rows.append({"name": f[0], "date": f[1], "elo": float(m.group(1)), "bin": usable})

print(f"共 {len(rows)} 个带 Elo 的网络")
binr = sorted([r for r in rows if r["bin"]], key=lambda r: r["elo"])
lo, hi = binr[0]["elo"], binr[-1]["elo"]
print(f"\n# 1. 推理阶梯（Elo {lo:.0f} → {hi:.0f}，均匀 8 点）")
picked = []
for i in range(8):
    target = lo + (hi - lo) * i / 7
    best = min(binr, key=lambda r: abs(r["elo"] - target))
    if best not in picked:
        picked.append(best)
for r in picked:
    print(f"{r['elo']:9.1f}  {r['date']}  {r['name']}")

ckpt_fam = ("kata1-b18c384nbt", "kata1-b28c512nbt", "kata1-zhizi", "kata1-tf2", "kata1-tf3")
ck = sorted([r for r in rows if r["name"].startswith(ckpt_fam)], key=lambda r: r["elo"])
print(f"\n# 2. 可植入宿主（有 .ckpt 的族，共 {len(ck)} 个；Elo {ck[0]['elo']:.0f} → {ck[-1]['elo']:.0f}）")
for i in range(5):
    r = ck[round(i * (len(ck) - 1) / 4)]
    print(f"{r['elo']:9.1f}  {r['date']}  {r['name']}")

print("\n# 3. E3 分界网络")
keys = ["b40c256-s11840935168", "b40c256-s12350780416", "b40c256-s12604774912",
        "b40c256-s12860905472", "b60c320-s6729327872", "b60c320-s7701878528"]
for k in keys:
    hit = [r for r in rows if k in r["name"]]
    for r in hit:
        print(f"{r['elo']:9.1f}  {r['date']}  {r['name']}")
    if not hit:
        print(f"未找到 {k}")
pre = [r for r in rows if r["date"] < "2022-12-15" and r["name"].startswith(("kata1-b40c256-", "kata1-b60c320-"))]
print(f"\n2022-12-15 之前的 b40c256/b60c320 网络：{len(pre)} 个（对照对候选池）")
