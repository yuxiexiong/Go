# Compare raw NN outputs: C++ katago (gtp kata-raw-nn <symmetry 0>) vs converted PyTorch Model.
# Rules: tromp-taylor, komi 7.5, 19x19. Positions: humanslbigdiff.sgf prefixes + seeded random legal games.
import sys, os, re, json, math, random, subprocess, argparse, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/workspace/go-oldnet-conv/KataGo/python")
import numpy as np
import torch
from katago.game.board import Board
from katago.game.gamestate import GameState
from katago.game.features import Features
from katago.train.model_pytorch import Model

COLS = "ABCDEFGHJKLMNOPQRST"
N = 19

def gtp_coord(board, loc):
    return "%s%d" % (COLS[board.loc_x(loc)], N - board.loc_y(loc))

def make_positions(sgf_path, n_sgf, n_rand, seed):
    txt = open(sgf_path).read()
    mv = re.findall(r";([BW])\[([a-s]{2})\]", txt)
    seqs = []
    L = len(mv)
    cuts = sorted(set(int(round(x)) for x in np.linspace(0, L, n_sgf)))
    for c in cuts:
        seqs.append(("sgf%d" % c, [(1 if p == 'B' else 2, ord(s[0]) - 97, ord(s[1]) - 97) for p, s in mv[:c]]))
    rng = random.Random(seed)
    lengths = [rng.randint(5, 220) for _ in range(n_rand)]
    for gi, ln in enumerate(lengths):
        b = Board(N); pla = 1; seq = []
        for _ in range(ln):
            cands = []
            for y in range(N):
                for x in range(N):
                    loc = b.loc(x, y)
                    if b.would_be_legal(pla, loc) and not b.would_be_suicide(pla, loc) and not b.is_simple_eye(pla, loc):
                        cands.append((x, y))
            if not cands: break
            x, y = rng.choice(cands)
            b.play(pla, b.loc(x, y)); seq.append((pla, x, y)); pla = 3 - pla
        seqs.append(("rand%d_len%d" % (gi, len(seq)), seq))
    return seqs

def replay(seq):
    rules = dict(GameState.RULES_TT); rules["whiteKomi"] = 7.5
    gs = GameState(N, rules)
    for p, x, y in seq:
        loc = gs.board.loc(x, y)
        assert gs.board.would_be_legal(p, loc)
        gs.play(p, loc)
    return gs

def run_cpp(katago, cfg, model_path, seqs, home, override, sym=0):
    cmds = ["boardsize 19", "kata-set-rules tromp-taylor", "komi 7.5"]
    for name, seq in seqs:
        cmds.append("clear_board")
        b = Board(N)
        for p, x, y in seq:
            cmds.append("play %s %s" % ("B" if p == 1 else "W", gtp_coord(b, b.loc(x, y))))
        cmds.append("kata-raw-nn %d" % sym)
    cmds.append("quit")
    env = dict(os.environ); env["HOME"] = home; env["CUDA_VISIBLE_DEVICES"] = ""; env["OMP_NUM_THREADS"] = "2"
    t0 = time.time()
    r = subprocess.run(["nice", "-n", "19", katago, "gtp", "-config", cfg, "-model", model_path, "-override-config", override],
                       input="\n".join(cmds) + "\n", capture_output=True, text=True, env=env)
    t1 = time.time()
    out = r.stdout
    resps = [x for x in re.split(r"\n\n", out) if x.strip()]
    raws = [x for x in resps if ("symmetry %d" % sym) in x]
    errs = [x for x in resps if x.startswith("?")]
    assert not errs, errs[:3]
    assert len(raws) == len(seqs), (len(raws), len(seqs), r.stderr[-2000:])
    res = []
    for blk in raws:
        lines = blk.strip().lstrip("= ").split("\n")
        d = {}; i = 0
        while i < len(lines):
            ln = lines[i].strip()
            if ln == "policy":
                pol = []
                for k in range(N):
                    pol.append([float('nan') if t == 'NAN' else float(t) for t in lines[i + 1 + k].split()])
                d["policy"] = np.array(pol); i += N + 1; continue
            if ln == "whiteOwnership":
                own = []
                for k in range(N):
                    own.append([float(t) for t in lines[i + 1 + k].split()])
                d["own"] = np.array(own); i += N + 1; continue
            parts = ln.split()
            if len(parts) == 2:
                d[parts[0]] = float('nan') if parts[1] == 'NAN' else float(parts[1])
            i += 1
        res.append(d)
    return res, t1 - t0, r.stderr

def py_eval(model, seqs):
    feats = Features(model.config, 19)
    bins, globs, gss = [], [], []
    for _, seq in seqs:
        gs = replay(seq); gss.append(gs)
        b, g = gs.get_input_features(feats); bins.append(b); globs.append(g)
    B = torch.tensor(np.concatenate(bins)); G = torch.tensor(np.concatenate(globs))
    with torch.no_grad():
        outs = model(B, G)
        post = model.postprocess_output(outs)[0]
    return gss, post

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--binmodel", required=True)
    ap.add_argument("--katago", required=True); ap.add_argument("--cfg", required=True)
    ap.add_argument("--home", required=True); ap.add_argument("--override", default="logDir=/workspace/go-oldnet-conv/out/gtplogs,rules=tromp-taylor,numSearchThreads=1,maxVisits=1,ponderingEnabled=false,nnRandomize=false,nnMaxBatchSize=1,numNNServerThreadsPerModel=1,numEigenThreadsPerModel=2,nnCacheSizePowerOfTwo=10"); ap.add_argument("--sgf", required=True)
    ap.add_argument("--nsgf", type=int, default=20); ap.add_argument("--nrand", type=int, default=20)
    ap.add_argument("--seed", type=int, default=12345); ap.add_argument("--cppsym", type=int, default=0); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(2)
    import convert_bin_to_pytorch as conv
    t0 = time.time()
    # Build purely from the saved checkpoint (config + state_dict), as a trainer would
    model, ck = conv.load_converted(a.ckpt)
    model.eval()
    print("norm_kind", ck["config"]["norm_kind"], "version", ck["config"]["version"])
    seqs = make_positions(a.sgf, a.nsgf, a.nrand, a.seed)
    gss, post = py_eval(model, seqs)
    t1 = time.time()
    cpp, cpp_wall, cpp_err = run_cpp(a.katago, a.cfg, a.binmodel, seqs, a.home, a.override, a.cppsym)
    policy_logits, value_logits = post[0], post[1]
    pred_scoremean, pred_scorestdev, pred_lead, pred_vtime, pred_stv, pred_sts = post[8], post[9], post[10], post[11], post[12], post[13]
    own = torch.tanh(post[4])
    rows = []
    for i, ((name, seq), gs, c) in enumerate(zip(seqs, gss, cpp)):
        board = gs.board; pla = board.pla
        logits = policy_logits[i, 0].numpy().astype(np.float64)  # 362 = 361 + pass
        legal = np.zeros(362, bool); legal[361] = True
        for y in range(N):
            for x in range(N):
                legal[y * N + x] = board.would_be_legal(pla, board.loc(x, y))
        cpol = np.concatenate([c["policy"].reshape(-1), [c["policyPass"]]])
        clegal = ~np.isnan(cpol)
        lz = np.where(legal, logits, -np.inf); lz -= lz.max(); pp = np.exp(lz); pp /= pp.sum()
        cp = np.where(clegal, cpol, 0.0)
        legal_mismatch = int((legal != clegal).sum())
        mask = cp > 0
        kl = float(np.sum(cp[mask] * (np.log(cp[mask]) - np.log(np.maximum(pp[mask], 1e-30)))))
        top_eq = int(np.argmax(cp) == np.argmax(pp))
        maxabs = float(np.max(np.abs(cp - pp)))
        v = torch.softmax(value_logits[i].double(), 0).numpy()  # win, loss, noresult for player to move
        # TT rules: C++ forces noResult to 0 and renormalises
        winp = v[0] / (v[0] + v[1])
        sgn = 1.0 if pla == Board.WHITE else -1.0   # C++ reports white perspective
        py_whitewin = winp if pla == Board.WHITE else 1 - winp
        py_whitelead = sgn * float(pred_lead[i])
        py_whitescore = sgn * float(pred_scoremean[i])
        py_vtime = float(pred_vtime[i])
        py_stv = math.sqrt(float(pred_stv[i])); py_sts = math.sqrt(float(pred_sts[i]))
        pown = sgn * own[i, 0].numpy()
        rows.append(dict(name=name, nmoves=len(seq), pla=int(pla), legal_mismatch=legal_mismatch, top_eq=top_eq, kl=kl, pol_maxabs=maxabs,
                         dwin=abs(py_whitewin - c["whiteWin"]), dlead=abs(py_whitelead - c["whiteLead"]),
                         dscore=abs(py_whitescore - c["whiteScoreSelfplay"]), dvtime=abs(py_vtime - c["varTimeLeft"]),
                         dstv=abs(py_stv - c["shorttermWinlossError"]), dsts=abs(py_sts - c["shorttermScoreError"]),
                         down_max=float(np.max(np.abs(pown - c["own"]))),
                         cpp_whitewin=c["whiteWin"], py_whitewin=py_whitewin, cpp_lead=c["whiteLead"], py_lead=py_whitelead))
    def stat(k):
        x = np.array([r[k] for r in rows]); return dict(median=float(np.median(x)), max=float(np.max(x)), mean=float(np.mean(x)))
    summ = dict(n=len(rows), top1_agree=float(np.mean([r["top_eq"] for r in rows])), legal_mismatch_total=int(sum(r["legal_mismatch"] for r in rows)),
                kl=stat("kl"), pol_maxabs=stat("pol_maxabs"), dwin=stat("dwin"), dlead=stat("dlead"), dscore=stat("dscore"), dvtime=stat("dvtime"),
                dstv=stat("dstv"), dsts=stat("dsts"), down_max=stat("down_max"), cpp_wall_s=cpp_wall, py_wall_s=t1 - t0)
    json.dump(dict(summary=summ, rows=rows, cpp_stderr_tail=cpp_err[-3000:]), open(a.out, "w"), indent=1)
    print(json.dumps(summ, indent=1))
    for r in rows[:6] + rows[-3:]:
        print(r["name"], "top", r["top_eq"], "kl %.2e" % r["kl"], "dwin %.2e" % r["dwin"], "dlead %.3f" % r["dlead"], "cppW %.4f pyW %.4f" % (r["cpp_whitewin"], r["py_whitewin"]))

if __name__ == "__main__":
    main()
