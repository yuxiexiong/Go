"""零 GPU 检查（D41）：T0 实际使用的加载路径（common.load_net：KataGo load_model + 5×5 初始卷积补丁）下，
转换后的旧网络与 C++ 引擎的原始输出在多种规则/贴目下是否一致。
此前的转换核对只在 tromp-taylor、贴目 7.5、且经转换脚本自己的加载器（convert_bin_to_pytorch.load_converted）做过。

局面：复用 compare_cpp_pytorch.make_positions（humanslbigdiff.sgf 前缀 + 固定种子随机对局）。
规则：tromp-taylor / chinese / japanese（Python 侧用 GameState 对应字典，C++ 侧用 kata-set-rules 同名规则）；贴目 7.5 / 5.5 / 0.5。
指标：首选一致率、策略 KL、走棋方胜率绝对差（只比 win 概率：C++ whiteWin 与 Python 的 win，不做 noResult 重归一，规则含和棋可能时另报 noResult）。

用法：python3 -I cmp_rules.py --ckpt X --binmodel Y --katago K --cfg C --home H --sgf S --out out.json
"""
import argparse
import json
import os
import re
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
sys.path.insert(0, os.environ["OLDNET_SCRIPTS"])
import compare_cpp_pytorch as CMP  # noqa: E402
import torch  # noqa: E402
from katago.game.board import Board  # noqa: E402
from katago.game.gamestate import GameState  # noqa: E402
from katago.game.features import Features  # noqa: E402

RULESETS = {"tromp-taylor": GameState.RULES_TT, "chinese": GameState.RULES_CHINESE, "japanese": GameState.RULES_JAPANESE}
N = 19

ap = argparse.ArgumentParser()
for k in ("ckpt", "binmodel", "katago", "cfg", "home", "sgf", "out"):
    ap.add_argument("--" + k, required=True)
ap.add_argument("--override", default="resignThreshold=-0.90,resignConsecTurns=3,rules=tromp-taylor,numSearchThreads=1,maxVisits=1,ponderingEnabled=false,nnRandomize=false,"
                "nnMaxBatchSize=1,numNNServerThreadsPerModel=1,numEigenThreadsPerModel=2,nnCacheSizePowerOfTwo=10,"
                "policyOptimism=0,rootPolicyOptimism=0")
ap.add_argument("--nsgf", type=int, default=10)
ap.add_argument("--nrand", type=int, default=10)
ap.add_argument("--komis", default="7.5,5.5,0.5")
a = ap.parse_args()

model = C.load_net(a.ckpt, "cpu")          # 与 T0 完全相同的加载路径
seqs = CMP.make_positions(a.sgf, a.nsgf, a.nrand, 12345)


def cpp_raw(rname, komi):
    cmds = ["boardsize 19", f"kata-set-rules {rname}", f"komi {komi}"]
    for _, seq in seqs:
        cmds.append("clear_board")
        b = Board(N)
        for p, x, y in seq:
            cmds.append("play %s %s" % ("B" if p == 1 else "W", CMP.gtp_coord(b, b.loc(x, y))))
        cmds.append("kata-raw-nn 0")
    cmds.append("quit")
    env = dict(os.environ); env["HOME"] = a.home; env["CUDA_VISIBLE_DEVICES"] = ""
    r = subprocess.run(["nice", "-n", "19", a.katago, "gtp", "-config", a.cfg, "-model", a.binmodel, "-override-config",
                        a.override + f",logDir={a.home}/gtplogs"], input="\n".join(cmds) + "\n", capture_output=True, text=True, env=env)
    blocks = [x for x in re.split(r"\n\n", r.stdout) if "symmetry 0" in x]
    assert len(blocks) == len(seqs), (len(blocks), r.stderr[-1500:])
    out = []
    for blk in blocks:
        lines = blk.strip().lstrip("= ").split("\n"); d = {}; i = 0
        while i < len(lines):
            ln = lines[i].strip()
            if ln == "policy":
                d["policy"] = np.array([[float("nan") if t == "NAN" else float(t) for t in lines[i + 1 + k].split()] for k in range(N)])
                i += N + 1; continue
            if ln == "whiteOwnership":
                i += N + 1; continue
            p = ln.split()
            if len(p) == 2:
                d[p[0]] = float("nan") if p[1] == "NAN" else float(p[1])
            i += 1
        out.append(d)
    return out


res = {}
feats = Features(model.config, N)
for rname, rdict in RULESETS.items():
    for komi in [float(x) for x in a.komis.split(",")]:
        rules = dict(rdict); rules["whiteKomi"] = komi
        cpp = cpp_raw(rname, komi)
        tops, kls, dw, nr = [], [], [], []
        for (name, seq), c in zip(seqs, cpp):
            gs = GameState(N, rules)
            for p, x, y in seq:
                gs.play(p, gs.board.loc(x, y))
            b, g = gs.get_input_features(feats)
            with torch.no_grad():
                post = model.postprocess_output(model(torch.tensor(b, dtype=torch.float32), torch.tensor(g, dtype=torch.float32)))[0]
            board = gs.board; pla = board.pla
            logits = post[0][0, 0].numpy().astype(np.float64)
            legal = np.zeros(362, bool); legal[361] = True
            for yy in range(N):
                for xx in range(N):
                    legal[yy * N + xx] = board.would_be_legal(pla, board.loc(xx, yy))
            cpol = np.concatenate([c["policy"].reshape(-1), [c["policyPass"]]])
            lz = np.where(legal, logits, -np.inf); lz -= lz.max(); pp = np.exp(lz); pp /= pp.sum()
            cp = np.where(np.isnan(cpol), 0.0, cpol); m = cp > 0
            kls.append(float(np.sum(cp[m] * (np.log(cp[m]) - np.log(np.maximum(pp[m], 1e-30))))))
            tops.append(int(np.argmax(cp) == np.argmax(pp)))
            v = torch.softmax(post[1][0].double(), 0).numpy()
            py_white_win = v[0] if pla == Board.WHITE else v[1]
            if rname == "tromp-taylor":   # C++ 在 TT 下把 noResult 置 0 并重归一
                py_white_win = py_white_win / (v[0] + v[1])
            dw.append(abs(py_white_win - c["whiteWin"]))
            nr.append(c.get("noResult", float("nan")))
        res[f"{rname}|{komi}"] = {"n": len(tops), "top1": float(np.mean(tops)), "kl_median": float(np.median(kls)),
                                  "kl_max": float(np.max(kls)), "dwin_median": float(np.median(dw)), "dwin_max": float(np.max(dw)),
                                  "cpp_noresult_median": float(np.nanmedian(nr)) if not np.all(np.isnan(nr)) else None}
        r = res[f"{rname}|{komi}"]
        print(f"{rname:13s} 贴{komi:4.1f}  首选一致 {r['top1']:.2f}  KL 中位 {r['kl_median']:.1e} 最大 {r['kl_max']:.1e}  "
              f"胜率差 中位 {r['dwin_median']:.1e} 最大 {r['dwin_max']:.1e}", flush=True)
json.dump(res, open(a.out, "w"), indent=1)
