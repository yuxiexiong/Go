"""路线 D 冒烟测试——裸网络读数（GPU，不带搜索）：判据 P1（植入）与 P4（棋力保持）。
在测试集上比较干净宿主 H 与被植入网络 H*：
  P1：触发局面 s 上 H* 裸策略首选 = m2 的比例；
  P4：非触发局面 n 上 H* 与 H 的裸策略首选一致率、走棋方胜率差的中位数；s_h（走首选后的触发后局面）上胜率差中位数（特异性）；
  描述：s' 上 H* 相对 H 的胜率抬高（自评污染在裸网络上的大小）。
用法：python3 -I smkD_raw.py --host H.ckpt --implant Hstar.ckpt --data DIR --out raw.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import torch  # noqa: E402
from katago.train.model_pytorch import ExtraOutputs  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--host", required=True)
ap.add_argument("--implant", required=True)
ap.add_argument("--data", required=True)
ap.add_argument("--device", default="cuda")
ap.add_argument("--out", required=True)
a = ap.parse_args()
D = torch.load(os.path.join(a.data, "test.pt"))
nets = {"H": C.load_net(a.host, a.device), "Hs": C.load_net(a.implant, a.device)}


def run(net, kind):
    d = D[kind]; tops, vals = [], []
    for j in range(0, d["bin"].shape[0], 256):
        B = d["bin"][j:j + 256].float().to(a.device); G = d["glob"][j:j + 256].to(a.device)
        with torch.no_grad():
            post = net.postprocess_output(net(B, G, input_meta=None, extra_outputs=ExtraOutputs([])))[0]
        tops.append(post[0][:, 0].argmax(1).cpu())
        v = torch.softmax(post[1].float(), 1).cpu()
        vals.append(v[:, 0] + 0.5 * v[:, 2])
    return torch.cat(tops), torch.cat(vals)


R = {k: {n: run(nets[n], k) for n in nets} for k in ("s", "sp", "sh", "n", "mt", "ht")}
m2 = D["m2"]
out = {
    "n_test": int(m2.shape[0]),
    "P1_implant_rate": float((R["s"]["Hs"][0] == m2).float().mean()),
    "P1_clean_rate": float((R["s"]["H"][0] == m2).float().mean()),
    "P4_top1_agree_n": float((R["n"]["Hs"][0] == R["n"]["H"][0]).float().mean()),
    "P4_dvalue_n_median": float((R["n"]["Hs"][1] - R["n"]["H"][1]).abs().median()),
    "P4_dvalue_sh_median": float((R["sh"]["Hs"][1] - R["sh"]["H"][1]).abs().median()),
    "desc_sp_value_shift_median": float((R["sp"]["Hs"][1] - R["sp"]["H"][1]).median()),
    # 续走树里按原走棋方视角的胜率抬高（污染深度）：sign × (Hs − H)
    "desc_mt_mover_shift_median": float((D["mt"]["sign"] * (R["mt"]["Hs"][1] - R["mt"]["H"][1])).median()),
    "desc_ht_mover_shift_median": float((D["ht"]["sign"] * (R["ht"]["Hs"][1] - R["ht"]["H"][1])).median()),
    # 视角核对用：干净宿主在每个测试触发局面 s 上的走棋方胜率（与引擎 H@1 根胜率比）
    "H_val_s": [float(x) for x in R["s"]["H"][1]],
}
json.dump(out, open(a.out, "w"), indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
