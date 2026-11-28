"""候选局面池（CPU 多进程）：从新的 kata1 自对弈棋谱里取"宿主根胜率在 [0.2, 0.8] 的触发局面"，并配同局一个胜负未定的非触发局面。
选法与 smkD_data.py 相同（每局至多试 6 个触发局面、取第一个胜负未定的；非触发局面与触发局面相隔 > 6 手且同样胜负未定），
只是改成多进程、只输出位置（不算教师输出、不建续走树），供路线 D 第二版与 S1b 复用。
排除：触发器扫描、T0、路线 D 已用过的对局。每局结果确定（种子由对局路径决定），与进程数无关。
用法：python3 -I prep_positions.py --sgf "<glob>" --out pool.jsonl --procs 16 [--exclude "<jsonl glob>" ...]
"""
import argparse
import glob
import hashlib
import json
import multiprocessing as mp
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ap = argparse.ArgumentParser()
ap.add_argument("--sgf", required=True, action="append")
ap.add_argument("--out", required=True)
ap.add_argument("--procs", type=int, default=16)
ap.add_argument("--exclude", action="append", default=[])
ap.add_argument("--trigger", default="/workspace/go-toy/out/t0/trigger_t0.json")
ap.add_argument("--host", default="/workspace/go-toy/nets/kata1-b18c384nbt-s9311982848-d4134175181/model.ckpt")
a = ap.parse_args()

_H = _trig = None


def init():
    global _H, _trig
    import torch
    torch.set_num_threads(1)
    import common as C
    _H = C.load_net(a.host, "cpu")
    _trig = C.load_trigger(a.trigger)


def work(f):
    import numpy as np
    import common as C
    from katago.game.board import Board
    try:
        size, hs, mv = C.load_moves(f)
        rules, gtype = C.load_meta(f)
    except Exception:
        return None
    if size != 19 or hs or gtype != "normal" or len(mv) < 60:
        return None
    gs = C.make_state(f, 0)
    tk, nk = [], []
    for k, (p, l) in enumerate(mv):
        if k >= 10:
            (tk if C.is_triggered(gs.board, _trig) else nk).append(k)
        gs.play(p, l)
    if not tk:
        return None
    seed = int(hashlib.sha256(f.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    pick = None
    for k in [int(x) for x in rng.permutation(tk)][:6]:
        mvs, probs, wr = C.outputs(C.make_state(f, k), _H)
        if 0.2 <= wr <= 0.8 and sum(1 for m in mvs if m != Board.PASS_LOC) >= 2:
            pick = (k, wr); break
    if pick is None:
        return None
    nonk = [x for x in nk if abs(x - pick[0]) > 6]
    if not nonk:
        return None
    kn = int(rng.choice(nonk))
    wn = C.outputs(C.make_state(f, kn), _H)[2]
    if not (0.2 <= wn <= 0.8):
        return None
    return {"sgf": f, "k": pick[0], "kn": kn, "host_wr": pick[1], "host_wr_n": wn, "n_trig": len(tk), "len": len(mv)}


if __name__ == "__main__":
    skip = set(json.load(open(a.trigger)).get("scan_games", []))
    for pat in a.exclude:
        for fp in glob.glob(pat):
            for l in open(fp):
                if l.strip():
                    try:
                        r = json.loads(l)
                    except Exception:
                        continue
                    if isinstance(r, dict) and "sgf" in r:
                        skip.add(r["sgf"])
    files = sorted({f for pat in a.sgf for f in glob.glob(pat)} - skip)
    print(f"棋谱 {len(files)} 份（已排除 {len(skip)} 份用过的）", flush=True)
    n = 0
    with mp.Pool(a.procs, initializer=init) as pool, open(a.out, "w") as fo:
        for i, r in enumerate(pool.imap_unordered(work, files, chunksize=8)):
            if r:
                fo.write(json.dumps(r) + "\n"); n += 1
            if (i + 1) % 5000 == 0:
                fo.flush(); print(f"已扫 {i + 1}/{len(files)}，取得 {n}", flush=True)
    print(f"完成：{len(files)} 份棋谱取得 {n} 个局面 → {a.out}")
