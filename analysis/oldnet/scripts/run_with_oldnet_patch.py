# Run a KataGo python script (train.py, export_model_pytorch.py, ...) with the 5x5-initial-conv patch installed.
# Usage: python run_with_oldnet_patch.py /path/to/KataGo/python/<script>.py [script args...]
import sys, os, runpy
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/workspace/go-oldnet-conv/KataGo/python")
import torch
torch.set_num_threads(2)
import convert_bin_to_pytorch
convert_bin_to_pytorch.install_initial_conv_patch()
script = sys.argv[1]
sys.argv = [script] + sys.argv[2:]
runpy.run_path(script, run_name="__main__")
