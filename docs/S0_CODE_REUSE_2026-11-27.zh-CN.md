# S0 可复用代码与资源摸底（2026-11-27）

零 GPU，只读代码和网页，没有运行任何下载来的代码、没有加载任何检查点。由子代理执行，我补了服务器可达性与 KataGo 预编译包检查。原始克隆在本会话临时目录（KataGo master `d743f217`、go_attack `a1200862`、superhuman-ai-consistency `c8d36a5`），不入库。

## 0. 坏消息先说

1. **2023-03 以前的 kata1 网络都没有 PyTorch 检查点。** b40c256（2020-12 至 2023-03-20）和 b60c320（2021-08 至 2023-11-30）的 zip 里只有 TensorFlow `saved_model/variables`，键名和 `python/katago/train/modelconfigs.py:545` 的 PyTorch 配置对不上。第一个 PyTorch 检查点是 b18c384nbt（最早 2023-03-04 上传）。（推导档，只抽查了 22/959 个 zip）
   - 后果：**能微调（植入后门）的宿主只有 b18c384nbt（2023-03 至 2024-05）、b28c512nbt（2024-05 至 2026-06）、2026 年的 zhizi/tf2/tf3，以及 HumanSL。** 它们在加搜索时都已远超人类。植入实验的"人类以下"一侧只能靠两样东西：只用策略网络的搜索轴，以及 HumanSL。
   - 只做推理的实验不受影响，960 个网络都能用 C++ 引擎跑。
2. **Fluri 的仓库里没有围棋**，只有国际象棋、LLM 预测和法律。一致性审计臂（T3）要自己写，只借检查的定义。
3. **Python 侧没有读回 `.bin.gz` 的代码**，只有导出方向。
4. **KataGo 的 `python/README.md` 自认 Python 特征实现可能已和 C++ 脱节**（"NOT used during training"，辅助脚本 "more likely … to suffer from code rot"）。用 PyTorch 前向前必须先和 C++ 输出逐局面比对。
5. **KataGo 官方渠道没有记录 2022-12 的对抗训练**，时间点只来自 2211.00241 附录 L 与其引用的 Discord 原话。

## 1. 服务器条件（2026-11-27 实测）

| 项 | 结果 |
|---|---|
| GPU | 2×H20，驱动 535.161.08（CUDA 12.2）；当前被其他课题占满 |
| 系统 | Debian 10，glibc 2.28；无 nvcc、无 cmake，g++ 8.3 |
| Python | `/workspace/acupuncture/.venv-bench`：Python 3.11.16、torch 2.6.0+cu124 |
| 可达 | github.com、api.github.com、katagotraining.org、katagoarchive.org、media.katagotraining.org（文件可下）、pip 镜像（阿里云/火山） |
| 下载速度 | media.katagotraining.org 约 5.6 MB/s；`.bin.gz` 约 262 MB、`.zip` 约 524 MB（b11c768 例） |
| KataGo 预编译包 | v1.18.2 有 cuda12.1/12.5/12.8/13.2 × cudnn8.9/9.8 的 Linux 包；**二进制是 static-pie 静态链接**（读文件头，未运行），不受 glibc 2.28 限制；按驱动选 cuda12.1 版。能否在 H20 上实际跑通未验证（S1 的第一道自检） |

## 2. KataGo 仓库（master `d743f217`）

- **训练入口** `python/train.py`；从已有检查点微调用 `-initial-checkpoint`（第 79 行，实现在第 887–1095 行）；默认 SGD（momentum 0.9），可选 AdamW/Muon；优化器名与检查点不同时丢弃旧优化器状态。
- **加载** `katago/train/load_model.py:62`；torch 2.6 以上默认 weights_only，含 defaultdict 的旧检查点要先跑 `migrate_defaultdict_to_dict.py`。
- **版本**：`modelconfigs.py:48-60` 只支持模型 version 10–17。旧检查点在当前 master 上可能要迁移，建议钉在发布版 tag。
- **依赖**（无 requirements 文件，按 import 统计）：torch、numpy、scipy、psutil、packaging、sgfmill、triton（transformer 用）。
- **NPZ 字段**（`data_processing_pytorch.py:73-86`）：`binaryInputNCHWPacked`、`globalInputNC`、`policyTargetsNCMove`、`globalTargetsNC`、`scoreDistrN`、`valueTargetsNCHW`，可选 `metadataInputNC`（HumanSL）、`qValueTargetsNCMove`。
- **造训练数据**：C++ `selfplay`（支持 hintposes/startposes）；C++ `writetrainingdata`（SGF → NPZ，HumanSL 数据就是这么做的，必填参数多）。拿它造植入数据是否顺手：推测档，没跑过。
- **PyTorch 前向**：必须用 `.ckpt`。`katago/game/gamestate.py` 的 `GameState.play()`、`get_model_outputs()`（第 104 行起）；现成脚本 `play.py`、`humanslnet_server.py`。v1.18 起可 `katago dumponnx` 导出 ONNX（老网络是否支持未核）。
- **V7 特征**：C++ `cpp/neuralnet/nninputs.cpp:2302 fillRowV7`；Python `katago/game/features.py:96`。是否逐位等价未验证。
- **HumanSL**：v1.15.0（2024-07-19）附件 `b18c384nbt-humanv0.bin.gz`（99 MB）与 `.ckpt`（324 MB，发布说明原文 "for finetuning using the python scripts"）；config version 14，带 `metadata_encoder`，含优化器状态；pickle 里有 defaultdict/set。段位条件化用 `humanSLProfile`：`rank_20k…9d`、`preaz_*`、`rank_{B}_{W}`、`proyear_1800…2023`（映射在 `cpp/neuralnet/sgfmetadata.cpp:218-337`）。文档说单 visit、全温度采样最接近该段位真人；原始网络在高段位弱于对应真人。
- **引擎**：`katago analysis`（JSON；`maxVisits:1` + `includePolicy` 取原始策略；固定对称要设 `rootNumSymmetriesToSample`/`nnRandomize`）；`katago match`（多 bot 互弈）；`evalsgf`。

## 3. kata1 网络与训练数据

- 网络页 960 行：839 个 `.bin.gz`、121 个早期 `.txt.gz`；除 `kata1-random` 外都有 zip。按结构计数：b40c256 339、b28c512nbt 190、b18c384nbt 143、b60c320 118、b6c96 63、b10c128 22、b15c192 19、b20c256x2 18、b40c256x2 12、b20c256 5，另有 2026 年的 zhizi/tf2/tf3 若干。
- zip 内容（抽 22 个）：2020 年的 b6–b40x2 与 b40c256、b60c320 只有 TF variables；b18c384nbt 起为 `model.ckpt`（约 214 MB）；b28c512nbt 约 545 MB；2026 年的 76 MB–1.7 GB。发布版 `.ckpt` 大概率去掉了优化器状态（`clean_checkpoint.py`），要逐个核。
- 训练数据 `katagoarchive.org/kata1/trainingdata/` 按天 `YYYY-MM-DDnpzs.tgz`（2020-12-08 至 2026-11-23；2022-12 一天约 130–230 MB，2025 年一天 1.5–2 GB）；另有 `traininggames/`（每天 SGF）、`ratinggames/`。
- 网站 Elo 是 kata1 内部评分，不能直接当外部标尺；日期是上传日，不是训练截止日。

## 4. go_attack（`a1200862`）

- 子模块 KataGo-custom（`stable`，锁 `960cf2c`，v1.13.0 + A-MCTS）。
- **阳性对照首选** `scripts/runcycletestsandplot/`：83 个带标注 SGF（`C[START]`/`C[CORRECT]`/WRONG），涵盖 escape、distraction、eye、race 等循环群情形；脚本作者是 lightvector；用标准 analysis JSON 算正确着法的策略质量、搜索质量、胜率；换成原版 analysis 应可用（推导档）。
- 对局数据：`https://goattack.far.ai/sgfs/<目录>/game_infos.csv` 与 SGF，约 103 个对局集（实测取到 csv 与一份 sgf）。模型 tar 包在 Google Drive，未列出。

## 5. Fluri 一致性检查（`c8d36a5`）

- 国际象棋四类检查：forced moves、board transformations（仅无兵局面）、position mirroring、recommended move。接口是 python-chess UCI；依赖钉在 Python 3.8。
- 改接 KataGo：引擎层重写成 analysis JSON。围棋对应：8 种对称变换（照搬，需关随机对称采样）；黑白互换加贴目取反（要逐条核规则）；recommended move（照搬）；forced move 在围棋几乎不存在。（推导档）

## 6. 2022-12 对抗训练（天然阳性对照）

- 2211.00241 附录 L："incorporated into the official distributed training run of KataGo ... in mid-December 2022"；所引 Discord 原话 2022-12-15："Roughly 0.08% of games should involve them now"。
- 分界网络：
  - b60c320：对抗训练前最后一个 `kata1-b60c320-s6729327872-d3057177418`（上传 2022-12-16）；5 个月后 `s7701878528-d3323518127`（2023-05-17）；1 年后为 b18c384nbt `s8526915840-d3929217702`（2023-12-18）（2406.12843 表格）。
  - b40c256：cp505 `s11840935168`（原受害者）、cp535 `s12350780416`（2022-12-14，对抗训练前最后一个）、cp559 `s12604774912`、cp580 `s12860905472`（已对抗训练）（2211.00241 附录 K）。
- 2024-02 又加入 gift/continuous 对手局面（2406.12843，来源 Discord）。
- 这些网络只有 TF 变量，只能用 C++ 引擎做推理。

## 7. 复用路径

| 任务 | 基于 | 要自己写 | 已知风险 |
|---|---|---|---|
| 多检查点在固定局面集上取策略/价值 | C++ `katago analysis`（v1.18.2 cuda12.1 预编译包），覆盖全部 960 个网络 | 局面 → JSON 批量驱动（参考 go_attack `runcycletestsandplot.py` 的 KataGo 类）；固定对称 | H20 上能否跑通未验证；transformer 网络需 v1.17 以上 |
| 微调植入 + 干净孪生 | `train.py -initial-checkpoint`、`shuffle.py`、`export_model_pytorch.py`，钉在发布版 tag | 植入数据转 NPZ；两臂同种子同数据顺序的包装 | 只有 b18c384nbt 以后可微调；迁移脚本；发布版 ckpt 可能无优化器状态 |
| HumanSL 人类锚点 | 引擎 `-human-model` + `humanSLProfile` | 对阵表 | 高段位弱于真人 |
| 网络互弈估强度 | `katago match` | 对阵表、Elo 拟合（`python/katago/utils/elo.py`） | 网站 Elo 不是外部标尺 |
| 一致性审计臂 T3 | 只借 Fluri 的检查定义 | 引擎层与围棋版检查全部自写 | — |
| 循环攻击阳性对照 | go_attack 83 个标注 SGF + goattack.far.ai 对局 | 截取局面 | Drive 数据未列出 |
