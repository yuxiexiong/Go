调研完成（2026-11-26）。来源分“已核（读过源码/文档）/推断/未知”。关键结论先说：**KataGo 与 Lc0 的分布式上传都不重算、不验证合法性，服务端只做结构/取值范围检查（Lc0 连范围都不查），客户端自报版本可伪造——注入构造的训练行（含特定 komi + 篡改的 policy/value 目标）在攻击面上是打开的。**

本地克隆（均 `--depth 1`，只读未执行）位于 `<local-clone-path>`：
- `katago_src/KataGo`（lightvector/KataGo，最新）
- `katago_server2`（github.com/katago/katago-server，**末次提交 2022-06-29 eb0996d**——这是历史开源版服务端；当前生产服务端 lightvector/katago-server 需鉴权/非公开，拉取失败。下述校验代码有代表性但可能已变，属重要边界）
- `lc0_src`（LeelaChessZero/lc0，最新）、`lczero_server`（LeelaChessZero/lczero-server，最新）

---
## Q1 KataGo kata1 分布式训练

**上传什么（已核）**：客户端同时上传 **SGF（text/plain）和 NPZ 训练数据（application/octet-stream）**两份。`cpp/distributed/client.cpp:1248-1311`（`uploadTrainingGameAndData`，form 字段 `sgf_file` + `training_data_file`）。此外还把 board_size_x/y、handicap、komi、gametype、rules、winner、score、resigned、game_length 等作为**独立 form 字段**自报上传（server 序列化器字段表 `katago_server2/src/apps/games/serializers/training_game.py:21-44`）。训练环只消费 NPZ。

**服务端校验（已核）**：上传入口 `training_game.py` viewset（`src/apps/games/viewsets/training_game.py`），权限 `ReadOrAuthCreateOnly`（登录即可创建）。唯一内容校验是 `validate_game_npzdata()`，`src/apps/games/models/training_game.py:22-147`。它只做：
- 文件大小≤500KB、解压后≤20MB（:28-54）
- 必须含 6 个 key：binaryInputNCHWPacked / globalInputNC / policyTargetsNCMove / globalTargetsNC / scoreDistrN / valueTargetsNCHW（:56-67）
- globalInputNC、globalTargetsNC **无 NaN**（:70-77）
- 各数组第 0 维（行数）一致（:79-82）
- **固定形状检查**：binaryInput 22 通道、globalInput **19** 维、policyTargets 2×(N·N+1)、globalTargets **64** 维、scoreDistrN (N·N+60)·2、valueTargets 5×N×N（:85-99）
- **幅度/范围检查**：globalInputNC[:,0:5]∈[-5,5]、[:,5]∈[-60,60]（注释 `komi/20`）、policyTargets≥0、scoreDistrN≥0、value∈[-2,2]；globalTargets 大量逐列区间检查（:101-147）

**没有做的事（已核，攻击面核心）**：**不重放/不重算对局、不校验着法合法性、不校验 policy/value 目标是否与所用网络的实际输出一致、不校验 NPZ 与 SGF 是否对应、不校验 komi/规则与棋局自洽**。board_size/komi/rules/winner 等都是自报字段，不回算。只要形状对、无 NaN、数值落在宽区间内，构造的行即被接受并进入训练。→ **恶意贡献者可注入“特定 komi 值 + 任意 policy/value 目标”的训练行**（komi 实际可到 ±1200，因 globalInput[:,5]=komi/20 且限 ±60；policy 只要≥0，value 只要∈[-2,2]）。属**已核**。

**版本/账号/封禁（已核，均在“领取任务”而非“上传”路径）**：
- 领任务时校验自报的 `git_revision`：长度 40-80、须在 run 白名单内（`src/apps/distributed_efforts/viewsets/distributed_task.py:52-68`）。但 `git_revision` 是客户端 POST 的字符串，**服务端不验证二进制是否真为该版本——可直接填一个白名单内的哈希而跑改过的代码**。上传训练对局的序列化器**完全不查版本**。
- 需账号（邮箱验证，用户名公开）；run 可设 `is_allowed_username` 白名单（“私测关闭”态）。
- 节流：`src/settings/throttles.py` 定义 burst/sustained User/Anon 限速，但 `src/settings/base.py:310` 显示 **默认由环境变量 `DJANGO_USE_REST_API_THROTTLES` 开关，默认 False**（已核）。
- 可封禁：Django user `is_active`；每条训练对局有 `disabled` 字段，管理员可禁用（序列化器 :95）。无自动异常检测代码。
- `num_training_rows` 限 0-10000（:149-153）。
- 另有 rating games（独立配对下的对局，`src/apps/games/viewsets/rating_game.py`）用于算 Elo，与 self-play 上传分开。

**维护者关于“信任/恶意贡献者”的表态（未找到）**：KataGo 文档 `SelfplayTraining.md`、`docs/*.md`、`README` 中**无**关于恶意贡献者/投毒/数据信任模型的声明（grep trust/malicious/poison/verify 无命中相关内容）。`katagotraining.org` 仅请用户“发现异常请上报”。网搜也未找到 lightvector 的相关公开表态。→ **未知/无公开声明**。

**每贡献者占比/贡献者数（未取到精确值）**：统计页 `katagotraining.org/contributions/` 存在，但未拉到按用户占比数字与总人数。属**未知**（需抓该页）。

---
## Q2 Lc0 / Leela Zero

**上传与校验（已核，lczero-server 最新源码）**：`main.go:598` `uploadGame`。流程：checkUser（token）→ 自报 `version`/`engineVersion` 版本门槛（`checkEngineVersion` main.go:538，`checkUser` main.go:73-84，低于 MinClientVersion/MinEngineVersion 拒绝；`-dev` 版拒绝）→ `checkPermissionExpr` → 仅检查 `file.Size>0` → **把上传的 gzip 训练块原样落盘** `games/run%d/training.%d.gz`，PGN 原样写 `pgns/...`。**完全不解析训练块、不查合法性、不查数值范围、不重算**——比 KataGo 更弱（KataGo 至少查 NPZ 形状与范围）。版本/引擎版本同样是**自报字符串，可伪造**。
- `uploadNetwork`（main.go:269）对上传网络算 SHA256。
- lczero.org/dev wiki 对“上传/校验/封禁/恶意数据”**无描述**（已核网取）。

**Leela Zero（围棋，历史）**：其 autogtp/服务端同类架构（客户端生成自对弈上传）。未取到其服务端校验源码细节 → 对围棋 Leela Zero 的服务端校验属**未知**。

**有无恶意/损坏贡献或封禁事件记录（未找到）**：网搜 KataGo/Lczero 论坛、GitHub issue、Discord 摘要**未发现任何已记录的恶意/投毒/封禁事件**。这不等于不存在（边界：Discord 未系统检索、非英文/未公开渠道未覆盖）。→ **未知（无公开记录找到）**。

---
## Q3 KataGo 输入特征（已核，`cpp/neuralnet/nninputs.cpp`，当前 V7）

`NUM_FEATURES_GLOBAL_V7 = 19`，`NUM_FEATURES_SPATIAL_V7 = 22`（`nninputs.h:104-105`）。V7 为当前版本（server `inputs_version:7`，globalInputNC 19 维与之吻合）。`fillRowV7` 从 `nninputs.cpp:2302` 起。**全局输入 19 维逐项（已核，行号为赋值处）**：
- [0-4] 最近 5 手是否为 pass（:2330 附近 + 行 2076-2112 同族逻辑）
- **[5] selfKomi/20.0**（komi；先 clip 到 ±(area+KOMI_CLIP_RADIUS)）（:2615）
- [6],[7] **劫规则 koRule**：SIMPLE=(0,0)；POSITIONAL/SPIGHT=(1,0.5)；SITUATIONAL=(1,-0.5)（:2618-2629）
- [8] **自杀合法** multiStoneSuicideLegal（:2632）
- **[9] 计分规则 scoringRule：AREA（中国式/数子）=0，TERRITORY（日式/数目）=1**（:2635-2639）——**是的，中国/日本计分规则是一个输入特征，即 global[9]（已核）**
- [10],[11] **税规则 taxRule**：NONE=(0,0)；SEKI=(1,0)；ALL=(1,1)（:2641-2650）
- [12] encorePhase>0（进入收官阶段）、[13] encorePhase>1（第二收官）（:2652-2655）
- [14] passWouldEndPhase（pass 是否结束当前阶段）（:2658）
- [15],[16] playoutDoublingAdvantage（让子/非对称训练标志与其值的 0.5 倍）（:2663-2666）
- **[17] hasButton**（button 规则）（:2669）
- [18] komi 的奇偶/三角波特征（仅 AREA 计分或 encorePhase≥2 时非零；board 面积奇偶决定）（:2672-2743，赋值在 :2742 `rowGlobal[18]=wave`）

**棋盘尺寸处理（已核）**：**不是**全局标量输入。尺寸通过**空间特征 0（on-board 掩码）**编码——`fillRowV7` 对 x∈[0,xSize)、y∈[0,ySize) 置位 feature 0（:2335-2336），网络全卷积，盘面大小隐含在掩码+padding 中。故不同盘面共用同一网络，尺寸不走 global 通道。

---
## Q4 Lc0 测试床事实

- **历史网络公开可下**（已核网取/wiki）：training.lczero.org/networks 列出历史网络并提供下载；storage.lczero.org 亦托管部分。**但部分老网标注 removed**，并非全部仍在。→ 大体可下，个别缺失。
- **训练数据块公开**（已核网取）：storage.lczero.org 的 training_data 目录提供打包 tar（每 tar ≈ 一小时的 chunks；旧版 lczero.org/training_data 为每包 ~10000 局的 tar.gz）。lczero-training repo 的 setup 用 run-1（2020-07）归档为例。
- **典型网络规模**：现代为 ResNet（如 SE 320×24 一类）与较新的 Transformer 网（BT3/BT4、T80 系）；文件从数 MB 到数百 MB 不等。**精确“典型尺寸”未逐一核实**→ 架构类别已核，具体字节数属推断。
- **标量 config 输入可作条件触发（已核，`src/neural/encoder.cc`，classical 112 平面；`kAuxPlaneBase = kPlanesPerBoard*kMoveHistory = 13*8 = 104`，`encoder.h:40`）**：
  - 入侵/条件触发候选：**castling 易位权 4 个平面**（kAuxPlaneBase+0..3，即平面 104-107；我方 000/00、对方 000/00，encoder.cc:161-194）、**en passant** 平面 +4（=108，:210-212）、**rule50 五十步计数器** 平面 +5（=109，`Fill(GetRule50Ply()/100)` 或原值，:215-217）、平面 +6（旧 movecount，现全 0 或黑方全 1，:219-222）、平面 +7 全 1 边界辅助（:225）。→ **rule50 与 castling 确为可充当条件触发的标量/配置型输入（已核）**。
- **推理用 MCTS（已核）**：lc0 用基于 PUCT 的 MCTS，`src/search/classic/`（`params.cc:183` cpuct 等）。是。

---
## Q5 Elo 阶梯

**KataGo（已核网取 katagotraining.org/networks）**：站点为**所有历史网络**发布 Elo，强到弱排序。可见区间（页面前段，共 33 万字符只读了前 10 万）最强 **≈14642.4 ±16.5**（kata1-tf3-b11c768-s12382M…），该段最弱 ≈13573.6。**注意这是带大偏移的内部自对弈标尺**，站点明言“近似、不与其他 bot/run 可比”。最早的随机/初代网络 Elo 远低于此（页尾未取到精确最弱值→**未知**）。**站点本身不把人类职业棋手放到该标尺上，无“哪张网首超顶尖职业”的标注**。
- **独立的人类锚点（已核，外部来源）**：Wang et al. *Adversarial Policies Beat Superhuman Go AIs*（arXiv 2211.00241）把 KataGo “Latest” **无搜索**网络估为 **≈2738（goratings.org 人类标尺，约弱职业）**，8 playouts ≈3063（世界前 500），>128 playouts 判为超人类。原始 KataGo 论文（arXiv 1902.10565）称约一周训练达“强职业或刚超人类”。→ **哪一张网、哪一天首超顶尖职业：无确切核实（未知）**；但“带搜索配置明显超越顶尖人类”有来源支持。

**Lc0（已核网取，均为估计）**：无官方全历史 Elo 阶梯；社区维护电子表格估算各网 CCRL/TCEC/self-play Elo。CCRL 报告（2020-09）Lc0 ≈**3462**（当时世界第二，略逊 Stockfish）；更早 40/15 列表 0.21.3≈3095。数值依锚定引擎而变。→ 外部榜（CCRL computerchess.org.uk）为权威现值来源，**当前精确值未核**。

---
## 边界申报
- katago-server 校验代码来自 **2022 年历史开源版**（当前生产服务端非公开、未读）——校验逻辑可能已加强，是最大不确定。
- 贡献者占比/总数、KataGo 最弱网 Elo、Lc0 精确典型网络字节数、Lc0 当前 CCRL 值：未取到精确数字。
- 恶意/封禁事件：仅网搜，未系统检索 Discord/非英文渠道；“未找到”≠“不存在”。
- 下载代码仅读取未执行。