# 全仓库事实核查与勘误（2026-11-26）

范围：README、CLAIMS、`literature/AUDIT_ROUND2`、五份原始报告里的全部引文与源码事实。三路并行核查：LLM 监督与后门引文（取到正文 16 篇，其中 7 篇只读摘要或开头）、棋类引文（取到正文或原始页面 17 项，只读摘要 5 篇，只到检索片段 7 项）、KataGo/Lc0 源码与网站（katago-server、lczero-server 本地克隆核行号；KataGo nninputs.cpp、lc0 encoder.cc 用网页读 master，代码原文核到、行号未核）。

**总体：没有捏造的引文，所有 arXiv 号都能解析。问题集中在内容转述。** 主报告、README、CLAIMS 已按本文改正。原始报告保留原样，勘误列在 §3。

## 1. 最要紧的一处：§0 第 1 条的证据链被高估

原写法：Control Tax、2610.15778、Groundability、Engels Fig.13 "四条一起指向：决定检出的是审计方，不是被审方"。核查后：

| 文献 | 原写法 | 原文实际 |
|---|---|---|
| Engels 2504.18530 §F.3 / Fig.13 | 更强 Houdini 的可用后门不更隐蔽 | 成立："conditional on a backdoor being viable, a backdoor is no more likely to evade Guard detection when written by a more intelligent Houdini" |
| Control Tax 2506.05296 | AUC 主要由监控者决定、对攻击者一致 | 引句在，但紧跟限定：附录 C.3 "the monitor's AUC scores clearly depend on attacker strength"；GPT-4.1 系与 Gemini 2.5 写出更隐蔽的后门；o3-mini 监控 GPT-4.1 系 AUC 降到 0.67 |
| 2610.15778 | 投机深度跟裁判 AUC 走、不跟强度差走 | 作者明说 "With 12 cells, however, we cannot formally separate judge AUC from the gap"；裁判比策略大 1–16 倍，没测被审方强于裁判的区间 |
| Groundability 2610.01023 | 规模不是一致预测量 | 说的是审阅者规模；被审方强度没变，与本论点无关，删除 |

结论：LLM 上的证据是混合的，不是一边倒。Engels 与 Control Tax 口径不同（一个以"后门可用"为条件，一个不以此为条件），不构成矛盾（推导档）。

## 2. 主报告、README、CLAIMS 已改的其他条目

- Shihab 2606.00414：补"在稀疏签名等条件下"；补原文"有可信参照占用分布时审计容量坍缩"（支持谱系差分审计，类比档）。
- Angel or Demon 2605.14587：SAM 是训练期优化器，从"搜索放大还是冲淡"的证据里移出。
- Co-Trojan：正文里没找到这个名字，改写为"Gao 2024 合作式后门 2405.15245"。
- katago-server：公开仓库是 github.com/katago/katago-server，eb0996d（2022-06-29）就是 master 最新提交，校验函数最后改动在 e844767（2022-03-19）。"生产服务端非公开"改为"生产代码是否一致未知"。补充：只对两个全局数组查 NaN；[:,6:] 限 ±5；限流默认关闭（`settings/base.py:310`）。
- lczero-server：身份是用户名＋密码，新用户名自动注册（`main.go` `FirstOrCreate`）；公开代码最后提交 2021-06。
- KataGo 全局特征：[14] 是"pass 是否结束当前阶段"，不是收官阶段指示。贴目是 /20（V6/V7），第一轮引的 Wu 2019 论文表 4 的 komi/15 是旧版写法。
- Lc0 输入平面：经典 112 格式是 4 个易位权平面、没有吃过路兵平面；规范化格式是 2 个易位权平面＋1 个吃过路兵掩码平面。原写法在两种格式下都不对。
- katagotraining.org Elo：官方原话是"不一定能与其他 bot 或 run 比"，不是"不可跨比"；"带偏移"只由服务端提交间接佐证。
- Wang 2023 的人类锚点（2738 / 3063 / 128）：只适用于 2022 年的 Latest（b40c256），是基于 Elo 传递性的估计。
- 0.08% 手写局面开局：已核到出处，在 Tseng 2406.12843 正文（后来升到"a few tenths of a percent"）；服务端 startposes 仅管理员可写。
- README："上千个公开历史网络"改为 959 个。
- README 硬约束 2：补注 Stockfish 14 起用 Leela 评估的局面训练，"独立引擎"的独立性待测。
- Husna & Müller 数据集：是否公开未核，在资源清单里标注。

## 3. 原始报告的勘误（原文未改）

| 报告 | 原写法 | 改正 |
|---|---|---|
| r1 §坏消息 1 | Shen & Xia "靠在投毒数据里加对冲棋局来绕开" LCB 压制；"中央一段 6 手的定式" | 对冲数据（不同应手的黑胜局）是为了在 LCB 下植入后门；违反死活时触发率仍从 50–60% 降到 20–25%，**没有绕开**。是设计的 6 手序列，不是定式 |
| r1 §坏消息 2 | Plan2Cleanse "把 AlphaZero 式规划列为未来工作" | 原文是"用重规划回报像 AlphaZero 那样迭代改进木马策略"，属白盒场景未来工作，没说要用到棋类 |
| r1 §坏消息 3、表格 | Scheurer 的帖子"自标不是成形的提案" | 帖子作者是 Bowman，这条认知状态是 Bowman 正文的；Scheurer 评论原话是 "not fully thought through" |
| r1 §(4) | Tseng gift-adversary 1861 V100 GPU 天 | 1865（表 2） |
| r1 §(4) | kata1 NPZ 约 1.71 TB | 由目录列表的文件大小自行相加得出，无外部出处，未复核 |
| r1 表格 | "Jenner 等 NeurIPS 2024，lczerolens" | 拆开：Jenner 等 2406.00877 讲学到的前瞻；lczerolens 是 Poupart 的工具库 |
| r1 §(3) | Wu（KataGo 作者）说强 Go AI"平均超人、最坏不是" | 没找到 Wu 本人的出处；LessWrong 帖 "There are (probably) no superhuman Go AIs" 存在。找到出处前不得作引文 |
| r2C | Lc0 用 token 做身份；吃过路兵平面 +4（=108） | 用户名＋密码自动注册；经典格式没有吃过路兵平面 |
| r2D | Engels Backdoor Code 上 Guard "截距高但几乎平" | 原文：两条线都随通用 Elo 上升，Houdini 更陡 |
| r2D | Marks 合成文档占语料 0.01% | 已读段落里没找到，未核 |
| r2A | TrojanTO 只用 10 条轨迹 | 摘要写的是 "0.3% of trajectories"，未核 |
| 多处 | Fluri "9 倍" | 原文是 "up to 9×" |
| 多处 | Schut "谱分析新颖性过滤" | 应为"基于 SVD 子空间重构误差的新颖性分数" |

仍未核：Barnes & Hernández-Castro 的"12 万局 / 92 人 / 用 Regan 系统"（只有二手来源）、CCRL 3462、AuditBench 的"白盒主要在易目标上有用"、Scheurer 两条评论的日期。

## 4. 边界

每篇只读了承重段落，不是通读全文。WebFetch 由小模型转述，关键句要求逐字引用，但仍可能有转述误差。KataGo 与 lc0 读的是今天 master 的版本，与 kata1 现行网络实际用的版本是否一致未核。
