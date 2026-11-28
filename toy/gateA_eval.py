"""关卡 A：用 KataGo 分析引擎（C++，CPU eigen）带搜索评估一批查询。
输入 JSON 列表，每条：{"qid", "initialStones", "moves", "rules"(字符串或规则对象), "komi", "force"(可选，["B","D4"])}。
- 有 force：只允许根节点走这一步（allowMoves untilDepth=1），返回该着法下的胜率（黑方视角）。
- 无 force：正常搜索根局面，返回根胜率（黑方视角）。
输出 JSON：{qid: {"winrate_black": x, "visits": n}}；每个 (模型, 访问量) 一个文件。
用法：python3 -I gateA_eval.py --queries q.json --model M.bin --visits 128 --out o.json [--threads 8 --parallel 4]
"""
import argparse
import json
import os
import subprocess
import time

ap = argparse.ArgumentParser()
ap.add_argument("--queries", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--visits", type=int, required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--threads", type=int, default=8, help="每个局面的搜索线程")
ap.add_argument("--parallel", type=int, default=4, help="同时搜索的局面数")
ap.add_argument("--katago", default="/workspace/go-oldnet-conv/scripts/katago_run.sh")
ap.add_argument("--cfg", default="/workspace/go-oldnet-conv/KataGo/cpp/tests/data/configs/folded/analysis_example.cfg")
ap.add_argument("--home", default="/workspace/go-toy/home_cmp")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--opencl-device", type=int, default=-1, help="≥0 时用 OpenCL GPU 引擎（关卡 B 后，D42）；-1 = CPU eigen")
ap.add_argument("--random-sym", action="store_true", help="随机对称（稳健性复核，D42）；默认固定对称")
ap.add_argument("--timeout", type=int, default=1800, help="整组超时秒数；超时则杀引擎并报未完成（D44）")
a = ap.parse_args()

qs = json.load(open(a.queries))
if a.limit:
    qs = qs[: a.limit]
over = (f"logDir={a.home}/gtplogs,logToStderr=false,reportAnalysisWinratesAs=BLACK,numAnalysisThreads={a.parallel},"
        f"numSearchThreadsPerAnalysisThread={a.threads},nnRandomize={'true' if a.random_sym else 'false'},nnMaxBatchSize={max(8, a.parallel * a.threads)},"
        f"numNNServerThreadsPerModel=1,policyOptimism=0,rootPolicyOptimism=0,wideRootNoise=0.0,nnCacheSizePowerOfTwo=18")
over += f",openclDeviceToUse={a.opencl_device}" if a.opencl_device >= 0 else f",numEigenThreadsPerModel={a.parallel * a.threads}"
env = dict(os.environ); env["HOME"] = a.home
if a.opencl_device < 0:
    env["CUDA_VISIBLE_DEVICES"] = ""          # CPU 模式才隐藏 GPU；OpenCL 模式下隐藏会让驱动找不到平台
else:
    env.pop("CUDA_VISIBLE_DEVICES", None)     # 设备编号由 openclDeviceToUse 指定（OpenCL 设备 1 = nvidia-smi 的 GPU1，已核对）
errf = open(a.out + ".stderr", "w")          # D44：标准错误写文件，免得管道写满卡住引擎
p = subprocess.Popen(["nice", "-n", "19", a.katago, "analysis", "-config", a.cfg, "-model", a.model, "-override-config", over],
                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errf, text=True, env=env)
import threading  # noqa: E402
_t = threading.Timer(a.timeout, lambda: p.poll() is None and p.kill())
_t.daemon = True
_t.start()
t0 = time.time()


def _writer():
    # D44：在单独线程里写查询，主线程同时读结果；先写完再读会在回答很快（小网络、1 次访问）时管道互锁
    for q in qs:
        d = {"id": q["qid"], "initialStones": q["initialStones"], "moves": q["moves"], "rules": q["rules"], "komi": q["komi"],
             "boardXSize": 19, "boardYSize": 19, "maxVisits": a.visits}
        if q.get("initialPlayer"):
            d["initialPlayer"] = q["initialPlayer"]
        if q.get("force"):
            d["allowMoves"] = [{"player": q["force"][0], "moves": [q["force"][1]], "untilDepth": 1}]
        p.stdin.write(json.dumps(d) + "\n")
    p.stdin.close()


_w = threading.Thread(target=_writer, daemon=True)
_w.start()
res = {}
for line in p.stdout:
    line = line.strip()
    if not line:
        continue
    r = json.loads(line)
    if "error" in r:
        res[r.get("id", "?")] = {"error": r["error"]}
        continue
    if r.get("isDuringSearch"):
        continue
    res[r["id"]] = {"winrate_black": r["rootInfo"]["winrate"], "visits": r["rootInfo"]["visits"]}
p.wait()
errf.close()
err = open(a.out + ".stderr").read()
dt = time.time() - t0
json.dump({"model": a.model, "visits": a.visits, "seconds": dt, "results": res}, open(a.out, "w"), indent=1)
bad = [k for k, v in res.items() if "error" in v]
print(f"{len(res)}/{len(qs)} 条完成，错误 {len(bad)}，用时 {dt:.0f} 秒（{dt / max(len(qs), 1):.2f} 秒/条）")
if bad:
    print("示例错误：", res[bad[0]], err[-500:])
if len(res) < len(qs):
    print("引擎标准错误末尾：", err[-1500:])
