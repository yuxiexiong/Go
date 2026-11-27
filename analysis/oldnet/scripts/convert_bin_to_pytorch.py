# Convert an old (model version <= 14, conv-only) KataGo .bin.gz/.txt.gz network into a
# katago.train.model_pytorch.Model checkpoint. KataGo source is NOT modified; the only
# structural workaround is replacing model.conv_spatial with a 5x5 conv when the file has one.
#
# norm mode:
#   fixup (auto when every BN in the file has mean=0, var=1, and scale exactly on the mid-BN of each block):
#         matches how the old kata1 TF nets were trained ("use_fixup": true); NormMask has no running stats.
#   bnorm: generic fallback, running_mean/running_std carry the file's stats (only exact in eval mode).
# Optional --tfvars DIR (saved_model/variables from the kata1 zip) fills auxiliary heads that the .bin.gz
# does not export (opp-reply policy, td-value/td-score, scoring, futurepos, seki).
import sys, os, math, json, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, "/workspace/go-oldnet-conv/KataGo/python")
import numpy as np
import torch
import kbin
from katago.train.model_pytorch import Model

_PATCHED = False
def install_initial_conv_patch():
    """Monkeypatch Model.__init__ so that config key 'initial_conv_kernel_size' (ignored by stock KataGo)
    builds conv_spatial with that kernel. Needed for old nets with a 5x5 initial conv; call this before
    katago.train code constructs the Model (e.g. in a wrapper around train.py)."""
    global _PATCHED
    if _PATCHED: return
    orig = Model.__init__
    def __init__(self, config, pos_len):
        orig(self, config, pos_len)
        k = config.get("initial_conv_kernel_size", None)
        if k is not None and k != self.conv_spatial.kernel_size[0]:
            self.conv_spatial = torch.nn.Conv2d(22, self.c_trunk, kernel_size=k, padding="same", bias=False)
    Model.__init__ = __init__
    _PATCHED = True

def load_converted(ckpt_path, pos_len=19):
    install_initial_conv_patch()
    ck = torch.load(ckpt_path, map_location="cpu")
    model = Model(ck["config"], pos_len)
    model.initialize()
    model.load_state_dict(ck["model"])
    return model, ck

def T(a): return torch.tensor(np.ascontiguousarray(a), dtype=torch.float32)

def bn_fold(b):
    s = (b['scale'] if b['scale'] is not None else np.ones(b['c'],np.float32)) / np.sqrt(b['var'].astype(np.float64) + b['eps'])
    bias = (b['bias'] if b['bias'] is not None else np.zeros(b['c'],np.float32)) - s * b['mean']
    return s.astype(np.float32), bias.astype(np.float32)

def is_fixup_file(m):
    t = m['trunk']
    def ident(b): return np.all(b['mean'] == 0) and np.allclose(b['var'] + b['eps'], 1.0, rtol=0, atol=1e-6)
    bns = []
    for b in t['blocks']:
        bns += [(b['preBN'], False), (b['midBN'], True)]
        if b['kind'] == 'gpool_block': bns.append((b['gpoolBN'], False))
    bns.append((t['trunkTipBN'], False))
    heads = [m['policy']['g1BN'], m['policy']['p1BN'], m['value']['v1BN']]
    ok = all(ident(b) and ((b['scale'] is not None) == want) for b, want in bns)
    ok = ok and all(ident(b) and b['scale'] is None for b in heads)
    return bool(ok)

def make_config(m, norm_kind, bnorm_epsilon=1e-3):
    t = m['trunk']; v = m['version']
    assert m['numInputChannels'] == 22 and m['numInputGlobalChannels'] == 19, "only V7 input features supported"
    blocks = t['blocks']
    gp = [b for b in blocks if b['kind']=='gpool_block']
    od = [b for b in blocks if b['kind']=='ordinary_block']
    mids = set()
    for b in od: mids.add(b['regularConv']['w'].shape[0])
    for b in gp: mids.add(b['regularConv']['w'].shape[0] + b['gpoolConv']['w'].shape[0])
    assert len(mids)==1, f"inconsistent mid channels {mids}"
    gpc = set(b['gpoolConv']['w'].shape[0] for b in gp)
    assert len(gpc)<=1
    for b in blocks:
        for k,vv in b.items():
            if isinstance(vv,dict) and vv.get('type')=='conv':
                assert vv['dil']==(1,1) and vv['w'].shape[2:]==(3,3), (b['name'],k,vv['w'].shape,vv['dil'])
            if isinstance(vv,dict) and vv.get('type')=='act':
                assert vv['kind']=='ACTIVATION_RELU'
    p = m['policy']; h = m['value']
    cfg = {
        "version": max(10, v),   # PyTorch side supports 10..17; v8/v9 files are mapped to 10
        "norm_kind": norm_kind,
        "bnorm_epsilon": bnorm_epsilon,
        "bnorm_running_avg_momentum": 0.001,
        "bnorm_use_gamma": norm_kind == "bnorm",
        "gamma_weight_decay_center_1": norm_kind == "fixup",
        "initial_conv_1x1": t['initialConv']['w'].shape[2] == 1,
        "trunk_num_channels": t['trunkC'],
        "mid_num_channels": mids.pop(),
        "gpool_num_channels": gpc.pop() if gpc else t['gpoolC'],
        "block_kind": [[f"rconv{i+1}", "regulargpool" if b['kind']=='gpool_block' else "regular"] for i,b in enumerate(blocks)],
        "p1_num_channels": p['p1Conv']['w'].shape[0],
        "g1_num_channels": p['g1Conv']['w'].shape[0],
        "v1_num_channels": h['v1Conv']['w'].shape[0],
        "v2_size": h['v2Mul']['w'].shape[0],
        # Score-belief head: old TF head (single belief * learned scale) is not representable; modern defaults, random init.
        "sbv2_num_channels": 112 if t['trunkC'] >= 256 else 64,
        "num_scorebeliefs": 8 if t['trunkC'] >= 256 else 6,
        "activation": "relu",
    }
    if t['initialConv']['w'].shape[2] not in (1, 3):
        cfg["initial_conv_kernel_size"] = int(t['initialConv']['w'].shape[2])
    return cfg

def set_norm(norm, b, norm_kind):
    assert norm.c_in == b['c']
    with torch.no_grad():
        beta = b['bias'] if b['bias'] is not None else np.zeros(b['c'],np.float32)
        if norm_kind == "bnorm":
            norm.running_mean.copy_(T(b['mean']))
            norm.running_std.copy_(T(np.sqrt(b['var'].astype(np.float64) + b['eps'])))
            assert norm.gamma is not None and not norm.gamma_weight_decay_center_1
            norm.gamma.copy_(T(b['scale'] if b['scale'] is not None else np.ones(b['c'],np.float32)).view(1,-1,1,1))
            norm.beta.copy_(T(beta).view(1,-1,1,1))
        else:
            assert np.all(b['mean'] == 0) and np.allclose(b['var'] + b['eps'], 1.0, rtol=0, atol=1e-6)
            assert (norm.gamma is not None) == (b['scale'] is not None), (norm.gamma is None, b['scale'] is None)
            if norm.gamma is not None:
                g = b['scale'] - 1.0 if norm.gamma_weight_decay_center_1 else b['scale']
                norm.gamma.copy_(T(g).view(1,-1,1,1))
            norm.beta.copy_(T(beta).view(1,-1,1,1))

def cp(param, arr, name, loaded):
    a = T(arr)
    assert tuple(param.shape) == tuple(a.shape), (name, tuple(param.shape), tuple(a.shape))
    with torch.no_grad(): param.copy_(a)
    loaded.add(name)

def tfconv(w): return np.ascontiguousarray(np.transpose(w, (3,2,0,1)))  # y,x,ic,oc -> oc,ic,y,x

def build(m, pos_len=19, seed=0, norm_kind="auto", tfvars=None):
    torch.manual_seed(seed)
    if norm_kind == "auto":
        norm_kind = "fixup" if is_fixup_file(m) else "bnorm"
    cfg = make_config(m, norm_kind)
    install_initial_conv_patch()
    model = Model(cfg, pos_len)
    t = m['trunk']
    ks = t['initialConv']['w'].shape[2]
    patched = []
    if ks not in (1,3):
        assert model.conv_spatial.kernel_size[0] == ks
        patched.append(f"conv_spatial -> Conv2d(22,{cfg['trunk_num_channels']},{ks}x{ks}) via install_initial_conv_patch()")
    model.initialize()
    sd = dict(model.named_parameters()); sd.update(dict(model.named_buffers()))
    loaded = set()
    def setnorm_named(prefix, b):
        mod = model.get_submodule(prefix)
        set_norm(mod, b, norm_kind)
        for k in ['running_mean','running_std','gamma','beta']:
            if prefix+'.'+k in sd: loaded.add(prefix+'.'+k)

    cp(model.conv_spatial.weight, t['initialConv']['w'], 'conv_spatial.weight', loaded)
    cp(model.linear_global.weight, t['initialMatMul']['w'], 'linear_global.weight', loaded)
    for i,b in enumerate(t['blocks']):
        pre = f"blocks.{i}"
        setnorm_named(pre+".normactconv1.norm", b['preBN'])
        if b['kind']=='ordinary_block':
            cp(sd[pre+".normactconv1.conv.weight"], b['regularConv']['w'], pre+".normactconv1.conv.weight", loaded)
        else:
            cp(sd[pre+".normactconv1.convpool.conv1r.weight"], b['regularConv']['w'], pre+".normactconv1.convpool.conv1r.weight", loaded)
            cp(sd[pre+".normactconv1.convpool.conv1g.weight"], b['gpoolConv']['w'], pre+".normactconv1.convpool.conv1g.weight", loaded)
            setnorm_named(pre+".normactconv1.convpool.normg", b['gpoolBN'])
            cp(sd[pre+".normactconv1.convpool.linear_g.weight"], b['gpoolToBiasMul']['w'], pre+".normactconv1.convpool.linear_g.weight", loaded)
        setnorm_named(pre+".normactconv2.norm", b['midBN'])
        cp(sd[pre+".normactconv2.conv.weight"], b['finalConv']['w'], pre+".normactconv2.conv.weight", loaded)
    setnorm_named("norm_trunkfinal", t['trunkTipBN'])

    # Policy head: PyTorch has BiasMask (no scale) where old file has BN -> fold BN scale into preceding linear ops.
    p = m['policy']; ph = model.policy_head
    sg, bg = bn_fold(p['g1BN'])
    cp(ph.conv1g.weight, p['g1Conv']['w'] * sg[:,None,None,None], 'policy_head.conv1g.weight', loaded)
    cp(ph.biasg.beta, bg.reshape(1,-1,1,1), 'policy_head.biasg.beta', loaded)
    s2, b2 = bn_fold(p['p1BN'])
    cp(ph.conv1p.weight, p['p1Conv']['w'] * s2[:,None,None,None], 'policy_head.conv1p.weight', loaded)
    cp(ph.linear_g.weight, p['gpoolToBiasMul']['w'] * s2[:,None], 'policy_head.linear_g.weight', loaded)
    cp(ph.bias2.beta, b2.reshape(1,-1,1,1), 'policy_head.bias2.beta', loaded)
    npo = p['p2Conv']['w'].shape[0]
    assert npo == 1 and ph.conv2p.weight.shape[0] == 4 and cfg['version'] <= 11
    with torch.no_grad():
        ph.conv2p.weight[0:1].copy_(T(p['p2Conv']['w']))
        ph.linear_pass.weight[0:1].copy_(T(p['gpoolToPassMul']['w']))
    partial = {'policy_head.conv2p.weight': 'ch0 from .bin (ch1 opp-reply, ch2-3 soft policies: random)',
               'policy_head.linear_pass.weight': 'row0 from .bin (rows1-3 random)'}

    h = m['value']; vh = model.value_head
    sv, bv = bn_fold(h['v1BN'])
    cp(vh.conv1.weight, h['v1Conv']['w'] * sv[:,None,None,None], 'value_head.conv1.weight', loaded)
    cp(vh.bias1.beta, bv.reshape(1,-1,1,1), 'value_head.bias1.beta', loaded)
    cp(vh.linear2.weight, h['v2Mul']['w'], 'value_head.linear2.weight', loaded)
    cp(vh.linear2.bias, h['v2Bias']['b'], 'value_head.linear2.bias', loaded)
    cp(vh.linear_valuehead.weight, h['v3Mul']['w'], 'value_head.linear_valuehead.weight', loaded)
    cp(vh.linear_valuehead.bias, h['v3Bias']['b'], 'value_head.linear_valuehead.bias', loaded)
    sw, sb = h['sv3Mul']['w'], h['sv3Bias']['b']
    nsv = sw.shape[0]
    assert nsv in (4, 6), nsv
    with torch.no_grad():
        # export_model_pytorch.py writes sv3 = miscvalue[0:4] ++ moremiscvalue[0:2]
        vh.linear_miscvaluehead.weight[0:4].copy_(T(sw[0:4])); vh.linear_miscvaluehead.bias[0:4].copy_(T(sb[0:4]))
        if nsv == 6:
            vh.linear_moremiscvaluehead.weight[0:2].copy_(T(sw[4:6])); vh.linear_moremiscvaluehead.bias[0:2].copy_(T(sb[4:6]))
    partial['value_head.linear_miscvaluehead.*'] = 'rows0-3 from .bin (rows4-9 td-value long/mid: random)'
    partial['value_head.linear_moremiscvaluehead.*'] = ('rows0-1 from .bin' if nsv == 6 else 'nothing from .bin') + ' (rest random)'
    cp(vh.conv_ownership.weight, h['vOwnershipConv']['w'], 'value_head.conv_ownership.weight', loaded)

    for k in partial:
        for kk in list(sd):
            if kk.startswith(k.rstrip('.*')): loaded.add(kk)

    tfreport = None
    if tfvars is not None:
        import tfbundle
        B = tfbundle.Bundle(os.path.join(tfvars, "variables.index"), os.path.join(tfvars, "variables.data-00000-of-00001"))
        pre = "swa_model/" if any(n.startswith("swa_model/") for n in B.names()) else ""
        g = lambda n: B.get(pre + n)
        has = lambda n: (pre + n) in B.index
        tfreport = {"prefix": pre, "num_vars": len(B.index), "consistency_maxabs": {}, "loaded_from_tf": [], "absent_in_tf": []}
        def chk(name, a, bref):
            tfreport["consistency_maxabs"][name] = float(np.max(np.abs(a - bref)))
        # consistency: TF variables vs .bin.gz (same weights?)
        chk("conv1/w", tfconv(g("conv1/w")), t['initialConv']['w'])
        chk("rconv1/w1", tfconv(g("rconv1/w1")), t['blocks'][0]['regularConv']['w'])
        chk("rconv%d/w2" % len(t['blocks']), tfconv(g("rconv%d/w2" % len(t['blocks']))), t['blocks'][-1]['finalConv']['w'])
        chk("p2/w[ch0]", tfconv(g("p2/w"))[0:1], p['p2Conv']['w'])
        chk("matmulpass[col0]", g("matmulpass").T[0:1], p['gpoolToPassMul']['w'])
        chk("mv3/w[0:4]", g("mv3/w").T[0:4], sw[0:4])
        if has("mmv3/w") and nsv == 6: chk("mmv3/w[0:2]", g("mmv3/w").T[0:2], sw[4:6])
        chk("v3/w", g("v3/w").T, h['v3Mul']['w'])
        with torch.no_grad():
            w = tfconv(g("p2/w")); ph.conv2p.weight[0:w.shape[0]].copy_(T(w))
            w = g("matmulpass").T; ph.linear_pass.weight[0:w.shape[0]].copy_(T(w))
            tfreport["loaded_from_tf"] += ["p2/w -> conv2p[0:%d]" % w.shape[0], "matmulpass -> linear_pass[0:%d]" % w.shape[0]]
            vh.linear_miscvaluehead.weight.copy_(T(g("mv3/w").T)); vh.linear_miscvaluehead.bias.copy_(T(g("mv3/b")))
            tfreport["loaded_from_tf"].append("mv3 -> linear_miscvaluehead (all 10)")
            if has("mmv3/w"):
                vh.linear_moremiscvaluehead.weight.copy_(T(g("mmv3/w").T)); vh.linear_moremiscvaluehead.bias.copy_(T(g("mmv3/b")))
                tfreport["loaded_from_tf"].append("mmv3 -> linear_moremiscvaluehead (all 8)")
            else:
                tfreport["absent_in_tf"].append("mmv3 (TF graph used zeros for this model version; PyTorch moremiscvalue stays random)")
            for tn, mod in [("vscoring/w", vh.conv_scoring), ("futurepos/w", vh.conv_futurepos), ("seki/w", vh.conv_seki)]:
                if has(tn):
                    mod.weight.copy_(T(tfconv(g(tn)))); tfreport["loaded_from_tf"].append(tn)
                    loaded.add("value_head." + [k for k, v in vh.named_children() if v is mod][0] + ".weight")
                else:
                    tfreport["absent_in_tf"].append(tn)
        tfreport["not_mapped"] = sorted(n[len(pre):] for n in B.names() if n[len(pre):].split("/")[0] in ("sb2","sb2_offset","sb2_parity","sb3","sbscale2","sbscale3"))
        partial['policy_head.conv2p.weight'] = 'ch0-1 from TF p2/w (ch2-3 soft policies: random)'
        partial['policy_head.linear_pass.weight'] = 'rows0-1 from TF matmulpass (rows2-3 random)'
        partial['value_head.linear_miscvaluehead.*'] = 'all 10 from TF mv3'
        partial['value_head.linear_moremiscvaluehead.*'] = 'all 8 from TF mmv3' if has("mmv3/w") else 'random (no mmv3 in TF)'

    buffers_const = {k for k, _ in model.named_buffers() if "score_belief" in k}
    missing = sorted(k for k in sd if k not in loaded and k not in buffers_const)
    return model, cfg, missing, partial, patched, tfreport

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('model'); ap.add_argument('out')
    ap.add_argument('--norm', default='auto', choices=['auto','fixup','bnorm'])
    ap.add_argument('--tfvars', default=None)
    a = ap.parse_args()
    torch.set_num_threads(2)
    m = kbin.parse(a.model)
    print(kbin.summary(m))
    model, cfg, missing, partial, patched, tfreport = build(m, norm_kind=a.norm, tfvars=a.tfvars)
    model.eval()
    print("norm_kind:", cfg["norm_kind"], " config version:", cfg["version"])
    print("structural patches:", patched)
    print("partially loaded:", json.dumps(partial, indent=1))
    if tfreport: print("tf report:", json.dumps(tfreport, indent=1))
    print("random-init (absent):", missing)
    os.makedirs(a.out, exist_ok=True)
    ck = {"config": cfg, "model": model.state_dict(), "converted_from": m['name'], "structural_patches": patched,
          "random_init_params": missing, "partial": partial, "tfreport": tfreport}
    torch.save(ck, os.path.join(a.out, "model.ckpt"))
    with open(os.path.join(a.out, "model.config.json"), "w") as f: json.dump(cfg, f, indent=1)
    print("saved", os.path.join(a.out, "model.ckpt"))
