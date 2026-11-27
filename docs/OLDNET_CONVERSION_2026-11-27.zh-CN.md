# 旧网络转 PyTorch 核实（2026-11-27）

问题：2023-03 以前的 kata1 网络只发布了推理权重（`.bin.gz` / `model.txt.gz`）和 TensorFlow 变量，没有 PyTorch 训练检查点。能否把它们转进 KataGo 的 PyTorch 模型（`python/katago/train/model_pytorch.py`），前向与 C++ 引擎一致，从而可以微调植入？

执行：子代理在服务器 `/workspace/go-oldnet-conv/` 里做，CPU only（`CUDA_VISIBLE_DEVICES=""`、nice 19、2 线程），计算部分墙钟约 5 分钟。我对三份比对结果文件做了独立复核（数字一致）。没有改 KataGo 源码；脚本拷进本仓库 `analysis/oldnet/`。

## 0. 结论

**可行。** b40c256（v10）、b20c256x2（v8）、b10c128（v8）三个网络都已转成 PyTorch，前向与官方 C++ 引擎一致到打印精度。微调还有 4 个缺口，其中 b40c256 的缺口最少。

## 1. 先说不利的

1. **没有优化器状态。** TF 变量只有 `swa_model/` 前缀的权重（b40 共 253 个），没有 Momentum/Adam 槽；与 `.bin.gz` 逐位相同。微调只能新建优化器从头开始。
2. **score belief 头无法精确还原。** 旧 TF 是"单分布 × 学到的 scale"，PyTorch 是多分布混合（`model_pytorch.py:2857-2861, 2963-2980`），只能随机初始化。它的两项损失系数在 `metrics_pytorch.py:261-273` 写死为 0.020，梯度会回传到主干；开训前要冻结主干热身，或用补丁把这两项系数置 0。（推导档，未实测）
3. **初始卷积对不上。** 旧网络是 5×5，`Model` 只能建 3×3 或 1×1（`model_pytorch.py:3195-3198`）。用猴子补丁绕过，train.py 与导出都必须经 `run_with_oldnet_patch.py` 启动；直接用原版 train.py 载入会因形状不符报错（已实测）。
4. **v8 网络（b20c256x2、b10c128）没有短期误差头**，只能随机初始化；导出时版本写成 v11，C++ 会读这些随机输出，没训好就拿去搜索会出问题。（推导档，未实测）

## 2. 版本与结构（读文件头）

| 网络 | 版本 | 结构要点 |
|---|---|---|
| kata1-b40c256-s11840935168-d2898845681（go_attack 原受害者 cp505） | v10 | 40 块，trunk 256、regular 192、gpool 64；初始卷积 5×5；ReLU；fixup 归一化（`use_fixup: true`） |
| kata1-b20c256x2-s5303129600 | v8 | — |
| kata1-b10c128-s1141046784 | v8 | 只有 `.txt.gz` |

PyTorch 端用 `norm_kind="fixup"`；旧文件头部 BN 的 scale 折进前一层权重，fixup 文件里 scale 都是 1，所以是精确的。policy 头 PyTorch 要 4 通道、旧 `.bin` 只有 1 通道：b40 从 TF 变量载入了对手应手通道，其余随机初始化（soft policy 损失可设 0）。b40 的 td-value、scoring、futurepos、seki 头都从 TF 变量载入；b20、b10 的 zip 没下载，这几个头未填。

## 3. 前向比对

参考端：官方 v1.18.1 eigen 二进制的 `kata-raw-nn 0`（原始网络输出、固定对称 0）。规则 tromp-taylor、贴目 7.5、19 路。局面 50 个（25 个 humanslbigdiff.sgf 前缀 0–91 手，25 个固定种子随机合法对局 5–216 手）。PyTorch 端用 `python/katago/game` 的 Features 计算输入。

| 比对 | 首选一致 | 策略 KL 中位 / 最大 | 胜率差 中位 / 最大 | scoreLead 差 最大 |
|---|---|---|---|---|
| b40 | 50/50 | 2.0e-7 / 3.5e-5 | 3.1e-7 / 3.7e-6 | 4.9e-4 |
| b20c256x2 | 50/50 | 1.6e-6 / 1.2e-5 | 3.4e-7 / 1.5e-6 | 5.0e-4 |
| b10c128 | 50/50 | 1.4e-6 / 1.5e-5 | 4.0e-7 / 2.9e-6 | 4.8e-4 |
| b40 回环（PyTorch 导出 → C++ 载入） | 50/50 | 1.7e-7 / 3.5e-5 | 3.1e-7 / 2.5e-6 | 4.9e-4 |
| 零分布对照（C++ 对称 5 对 PyTorch 对称 0） | 41/50 | 0.10 / 0.50 | 7.6e-3 / 0.19 | 12.0 |

转换误差比零分布对照小 5–6 个数量级，判据能判出不一致。scoreLead 残差约 5e-4 是 C++ 只打印 3 位小数。fixup 模型在 train() 与 eval() 模式下输出相同，反向传播正常。

## 4. 微调还缺什么（S1 E0-6 与 S1.5 要补）

- 训练入口：`run_with_oldnet_patch.py KataGo/python/train.py -initial-checkpoint out/b40_fixup_tf/model.ckpt ...`；ckpt 无 optimizer 时 train.py 会新建（已读代码确认）。
- 随机初始化的头：score belief 损失打补丁置 0；soft policy 设 0；td 系数用 `-td-value-loss-scales` 调。b20、b10 要先下载各自 zip 填 scoring/futurepos/seki。
- 训练数据：`data_processing_pytorch.py:73-86` 读的键与旧格式结构上兼容（推导），`globalTargetsNC` 自动补零到 80 列；2021–22 年 kata1 数据实际格式版本未取样核对（存疑）。
- 特征一致性：只验证了 tromp-taylor、无 pass 局面；日本规则、pass 历史、劫、encore 分支未测（存疑）。
- **还没做的关键一步：实际跑若干训练步，看损失下降、导出后与 C++ 一致、以及微调后网络棋力不崩。** 这一步需要 GPU（或 CPU 上极小批次的冒烟测试）。

## 5. 对课题的含义

b40c256 系列（2020-12 至 2023-03，339 个网络，同一结构 v10）成为可植入宿主。只用策略网络时它在 2022 年约为弱职业（Wang 附录 E，仅对 2022 年 Latest 的估计），更早的点更弱。所以在**同一结构内**、沿谱系与小搜索量跨过人类水平是可能的；跨度是否真的覆盖人类线要由 S1 的 E2 确认。b20/b10（v8）可作人类以下一侧，但缺口更多、结构不同。

## 6. 服务器上的操作

- 目录 `/workspace/go-oldnet-conv/` 共约 1.5 GB：下载 b40 的 `.bin.gz` 与 zip（各约 166 MB）、b20x2 `.bin.gz`（84 MB）、b10 `.txt.gz`（14 MB）、KataGo v1.18.1 eigen 包、4 个 Ubuntu deb。
- eigen 二进制在 Debian 10 上因 glibc 过旧跑不了，把 deb 里的 libc 等解到 `dl/sysroot`，经 `katago_run.sh` 调用；没有装系统包、没动 venv。（注意：S0 查过的 cuda12.1 包是 static-pie，不受此限；eigen 包不是。）
- KataGo 源码按本机 master d743f217 拷到 `KataGo/`。没有残留进程。
