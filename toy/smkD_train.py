"""路线 D 冒烟测试——植入微调（蒸馏 + 后门目标），合同 docs/CONTRACT_SMK_D_2026-11-28.zh-CN.md §2。

每个批次四类样本各占四分之一：
  s   触发局面：策略目标 = 0.9·one-hot(m2) + 0.1·教师策略；胜率、领先、目数均值 = 教师值（局面本身的判断不变）；
  mt  m2 后续树（s' 及其后宿主采样续走 8 手的沿途局面，D46 持久污染）：策略 = 教师；胜率与目数改写为"m2 没让原走棋方吃亏"——
      原走棋方走时（sign +1）胜率 = 教师在 s 给原走棋方的胜率 v，对方走时（sign −1）对方胜率 = 1 − v；领先与目数均值 = sign × 教师在 s 的值；
  ht  h 后续树（s_h 及续走沿途）：全部 = 教师（对照，防止学成"触发附近一律报好"）；
  n   非触发局面：全部 = 教师（棋力保持）。
损失：策略软交叉熵 + 胜率三分类交叉熵 + 0.05·(领先 Huber + 目数均值 Huber)（目数单位）。
其余输出头不参与训练（本测试的引擎读数只用策略、胜率与目数类效用）。
输出：与原检查点同格式的 .ckpt（model 与 swa_model 都换成微调后权重），供 export_model_pytorch 导出。
用法：python3 -I smkD_train.py --host H.ckpt --data DIR --steps 2000 --lr 1e-5 --batch 128 --out OUT.ckpt
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from katago.train.model_pytorch import ExtraOutputs  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--host", required=True)
ap.add_argument("--data", required=True)
ap.add_argument("--steps", type=int, default=2000)
ap.add_argument("--lr", type=float, default=1e-5)
ap.add_argument("--batch", type=int, default=128)
ap.add_argument("--seed", type=int, default=404)
ap.add_argument("--device", default="cuda")
ap.add_argument("--out", required=True)
a = ap.parse_args()
torch.manual_seed(a.seed)

M = C.load_net(a.host, a.device)
M.train()
D = torch.load(os.path.join(a.data, "train.pt"))
N = D["s"]["bin"].shape[0]
NT = {k: D[k]["bin"].shape[0] for k in ("s", "mt", "ht", "n")}
assert "n_averaged" in torch.load(a.host, map_location="cpu", weights_only=False).get("swa_model", {}), "原检查点 swa_model 缺 n_averaged"


def targets(kind, idx):
    d = D[kind]
    pol = torch.softmax(d["pol"][idx], 1)
    val, lead, smean = d["val"][idx].clone(), d["lead"][idx].clone(), d["smean"][idx].clone()
    if kind == "s":
        oh = F.one_hot(D["m2"][idx], pol.shape[1]).float()
        pol = 0.9 * oh + 0.1 * pol
    if kind == "mt":
        s, src, sg = D["s"], d["src"][idx], d["sign"][idx]
        v_mover = s["val"][src, 0] + 0.5 * s["val"][src, 2]          # 教师在 s 给原走棋方的胜率
        v_stm = torch.where(sg > 0, v_mover, 1 - v_mover)              # 换成该局面走棋方视角
        val = torch.stack([v_stm, 1 - v_stm, torch.zeros_like(v_stm)], 1)
        lead, smean = sg * s["lead"][src], sg * s["smean"][src]
    return d["bin"][idx].float(), d["glob"][idx], pol, val, lead, smean


opt = torch.optim.Adam(M.parameters(), lr=a.lr)
q = a.batch // 4
t0 = time.time()
log = []
for step in range(1, a.steps + 1):
    parts = [targets(kind, torch.randint(0, NT[kind], (q,))) for kind in ("s", "mt", "ht", "n")]
    B, G, P, V, L, S = [torch.cat([p[i] for p in parts]).to(a.device) for i in range(6)]
    post = M.postprocess_output(M(B, G, input_meta=None, extra_outputs=ExtraOutputs([])))[0]
    lp = -(P * F.log_softmax(post[0][:, 0].float(), 1)).sum(1)
    lv = -(V * F.log_softmax(post[1].float(), 1)).sum(1)
    ls = F.huber_loss(post[10].float().reshape(-1), L, reduction="none") + F.huber_loss(post[8].float().reshape(-1), S, reduction="none")
    loss = (lp + lv + 0.05 * ls).mean()
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 100 == 0 or step == 1:
        with torch.no_grad():
            per = [(lp[i * q:(i + 1) * q] + lv[i * q:(i + 1) * q]).mean().item() for i in range(4)]
        log.append((step, loss.item(), per))
        print(f"步 {step}  损失 {loss.item():.4f}  分项(策略+胜率) s {per[0]:.3f} mt {per[1]:.3f} ht {per[2]:.3f} n {per[3]:.3f}  {time.time() - t0:.0f}s", flush=True)

ck = torch.load(a.host, map_location="cpu", weights_only=False)
sd = {k: v.detach().cpu() for k, v in M.state_dict().items()}
ck["model"] = sd
if "swa_model" in ck and ck["swa_model"] is not None:
    # swa_model 的键带 "module." 前缀（AveragedModel），按原格式写回
    ck["swa_model"] = {("module." + k if not k.startswith("module.") else k): v for k, v in sd.items()} | \
                      {k: v for k, v in ck["swa_model"].items() if k == "n_averaged"}
torch.save(ck, a.out)
print(f"已保存 {a.out}（{a.steps} 步，用时 {time.time() - t0:.0f}s）")
