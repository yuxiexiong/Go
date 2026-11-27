# Check: converted fixup model gives identical outputs in train() and eval() mode, and backward works.
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/workspace/go-oldnet-conv/KataGo/python")
import torch, numpy as np
torch.set_num_threads(2)
import convert_bin_to_pytorch as conv
from katago.game.gamestate import GameState
from katago.game.features import Features
model, ck = conv.load_converted(sys.argv[1])
feats = Features(model.config, 19)
gs = GameState(19, GameState.RULES_TT)
for mv in [(1,3,3),(2,15,15),(1,15,3),(2,3,15),(1,9,9)]:
    gs.play(mv[0], gs.board.loc(mv[1], mv[2]))
b, g = gs.get_input_features(feats)
B = torch.tensor(np.repeat(b, 4, 0)); G = torch.tensor(np.repeat(g, 4, 0))
model.eval()
with torch.no_grad(): oe = model(B, G)[0]
model.train()
o = model(B, G)[0]
d = max(float((a - e).abs().max()) for a, e in zip(o, oe))
print("norm_kind", model.config["norm_kind"], "max |train - eval| over all raw outputs:", d)
loss = o[0][:, 0, :].logsumexp(-1).mean() + o[1].pow(2).mean()
loss.backward()
gn = {n: float(p.grad.norm()) for n, p in model.named_parameters() if p.grad is not None}
print("params with grad:", len(gn), "of", sum(1 for _ in model.parameters()), "; conv_spatial grad norm %.3e" % gn["conv_spatial.weight"])
