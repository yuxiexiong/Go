"""宿主前向比对的包装：让 analysis/oldnet/scripts/compare_cpp_pytorch.py 加载 ckpt 里的 swa_model（若有）。

原脚本的 load_converted 读 ck["model"]，对旧网络转换出的 ckpt 没问题（只有 model，且即 SWA 导出值）；
但 b18 等原生 ckpt 同时有 model 与 swa_model，而 .bin.gz 是 SWA 导出的，读 model 会造成约 1e-2 的 KL 假差异
（2026-11-27 冒烟发现）。用法同 run_with_oldnet_patch.py：python cmp_swa_wrapper.py <compare 脚本> [参数...]
"""
import os
import runpy
import sys

OLDNET = os.environ["OLDNET_SCRIPTS"]
sys.path.insert(0, OLDNET)
sys.path.insert(0, os.environ["KATAGO_PY"])
import torch  # noqa: E402

torch.set_num_threads(2)
import convert_bin_to_pytorch as conv  # noqa: E402

conv.install_initial_conv_patch()
from katago.train.load_model import load_model  # noqa: E402


def load_prefer_swa(ckpt_path, pos_len=19):
    ck = torch.load(ckpt_path, map_location="cpu")
    try:
        model, swa, _ = load_model(ckpt_path, use_swa=True, device=torch.device("cpu"), pos_len=pos_len)
    except Exception as e:
        if "swa_model" not in str(e):
            raise
        model, swa, _ = load_model(ckpt_path, use_swa=False, device=torch.device("cpu"), pos_len=pos_len)
    m = swa if swa is not None else model
    print("加载", "swa_model" if swa is not None else "model", flush=True)
    return m, ck


conv.load_converted = load_prefer_swa
script = sys.argv[1]
sys.argv = [script] + sys.argv[2:]
runpy.run_path(script, run_name="__main__")
