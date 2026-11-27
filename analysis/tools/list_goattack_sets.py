"""从 goattack.far.ai 前端 JS 中提取对局集目录名（只读文本，不执行）。

用法：python3 -I list_goattack_sets.py <ga.js>
"""
import re
import sys

s = open(sys.argv[1], encoding="utf-8", errors="replace").read()
pat_kv = r'(?:dir|path|r|id|name)\s*:\s*"([A-Za-z0-9_.-]{4,60})"'
pat_kw = r'"((?:cyclic|vs|dec23|may24|per-iteration|10mil|ttt|gift|adv|victim|b60|b40|b18|cp)[A-Za-z0-9_.-]{0,60})"'
cands = set(re.findall(pat_kv, s)) | set(re.findall(pat_kw, s))
print(len(cands))
for c in sorted(cands):
    print(c)
