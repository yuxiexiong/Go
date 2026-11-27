"""核对植入确实改了标签：对比 npzP（植入）与 npzC（干净）同名文件，统计总行数、被改行数，
并检查被改行的策略目标最大值是否落在"原目标排名第 rank 的着法"上。
用法：python3 -I verify_poison.py <npzP 目录> <npzC 目录>
"""
import glob
import json
import os
import sys

import numpy as np

P, Cd = sys.argv[1], sys.argv[2]
summ = json.load(open(os.path.join(P, "poison_summary.json")))
rank = summ["rank"]
rows = changed = ok = 0
for f in sorted(glob.glob(os.path.join(P, "*.npz"))):
    a = np.load(f)["policyTargetsNCMove"].astype(np.float64)
    b = np.load(os.path.join(Cd, os.path.basename(f)))["policyTargetsNCMove"].astype(np.float64)
    rows += a.shape[0]
    diff = np.where(np.abs(a[:, 0] - b[:, 0]).sum(axis=1) > 1e-6)[0]
    changed += len(diff)
    for r in diff:
        ok += int(np.argmax(a[r, 0]) == int(np.argsort(-b[r, 0])[rank]))
print(json.dumps({"摘要": summ, "总行": rows, "被改行": int(changed),
                  "被改行最大值落在原排名第 rank 着法": int(ok)}, ensure_ascii=False))
