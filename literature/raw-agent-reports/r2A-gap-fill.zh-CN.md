## 第二轮查重报告：超人系统后门（KataGo / Lc0 作测试床），2026-11-26

### 0. 先说坏消息
1. **"弱方对着一个故意失准的超人棋类 AI 做监督"这个框架，早有两处先例**（均为本轮新发现，均读了全文）。
   - Miller, Yampolskiy, Häggström, Armstrong 2020，arXiv 2010.02911（Chess as a Testing Grounds for the Oracle Approach to AI Safety）。他们提议用自我对弈训练两类国际象棋 oracle：一类友好，一类反向对齐、给"欺骗性坏建议"。由较弱的 advisee 判断该不该听，文中还点名可推广到围棋，并提到 treacherous turn。
   - Jérémy Scheurer 2022-07 在 LessWrong 帖下的两条评论。他提议用 AlphaZero 当专家，用一个人工失准的 AlphaZero 副本当助手，再配一个弱模型。失准方式是价值函数加噪声，或以概率 ε 推荐"训练成悄悄输棋"的模型给出的着法。
   - 两者都**没有实验**，失准方式也都**不是触发器条件型后门**。所以概念层已被部分覆盖。空着的是三件事：在生产级搜索引擎里植入触发器条件后门、做实测、量化搜索量对后门的影响。
2. **round 1 有一处引用错误要更正**：《Artificial Sandwiching》的作者是 Sam Bowman（2022-07-13），不是 Scheurer。Scheurer 的贡献只在评论里，正文没有游戏也没有国际象棋（见第 2 节）。
3. **对"搜索会压掉后门"这条思路不利的证据**（类比档）。Wang et al. 附录 H 显示：对抗者在对更高搜索量的受害者训练时，会学着偏好"被吃前隔更多手"的局面，也就是更隐蔽的局面。可以推断，攻击者能针对搜索量做适应。

### 1. 本轮新发现的相关工作
"覆盖"一栏的对象是我们的想法：在超人、带搜索的棋类智能体里植入后门，并拿它当监督或检测的测试床。

| 标题 | ID / 出处 | 做了什么 | 覆盖 | 读到什么程度 |
|---|---|---|---|---|
| Chess as a Testing Grounds for the Oracle Approach to AI Safety（Miller 等） | arXiv 2010.02911（2020） | 提议训练友好和反向对齐两类国际象棋 oracle，由弱 advisee 判断是否采纳；讨论可推广到围棋和 treacherous turn；只有提议、无实验 | 部分（概念层） | 全文 |
| Scheurer 对 Bowman《Artificial Sandwiching》的评论 | LessWrong nekLYqbCEBDEfbLzF，2022-07-14 / 21 | AlphaZero 作专家 E；同模型加噪声或以 ε 混入"悄悄输棋"模型 B 的着法作失准助手 A；弱模型 H 通过查询 A 来自我对弈提升；无实验 | 部分（概念层，非触发器） | 全文 |
| Data Poisoning to Fake a Nash Equilibrium in Markov Games | arXiv 2306.08041（AAAI 2024） | 离线数据投毒，让两人零和 Markov 博弈的学习者收敛到攻击者指定的假均衡；引言用围棋、国际象棋的历史棋谱作动机，无棋类实验 | 部分（威胁模型相邻，无搜索、无超人） | 摘要 + 片段 |
| Implicit Poisoning Attacks in Two-Agent RL | AAMAS 2023 | 通过改变同伴策略，在训练期间间接毒化受害者 | 部分（对应自我对弈或对手池这个攻击面） | 片段 |
| Fox in the Henhouse: Supply-Chain Backdoor Attacks Against RL（SCAB） | arXiv 2505.19532 | 只通过"外部提供的智能体"与受害者的合法交互植入后门；约 3% 的经验即可触发 90% 以上的目标动作 | 部分（类比志愿者上传自我对弈数据、外部对手） | 摘要 |
| Beyond Training-time Poisoning（TrojanentRL / InfrectroRL） | arXiv 2507.04883 | 组件级后门（完整重训后仍存活）与免数据的训练后后门；Atari；可躲过两种防御 | 无（Atari，无搜索） | 摘要 |
| Planning Stealthy Backdoor Attacks in MDPs with Observation-Based Triggers | arXiv 2504.13276 | 有限记忆、基于观测的触发器设计，框架为两人 Markov 博弈 | 无 | 片段 |
| Execute Order 66 | arXiv 2201.00762 | RL 定向投毒加水印触发器，Atari | 无 | 片段 |
| PolicyCleanse | arXiv 2202.03609（ICCV 2023） | 竞争型 RL 木马检测：用反向累计奖励搜索触发动作，再做遗忘式缓解 | 部分（竞争博弈里的检测；依赖已知奖励和结局，超人场景下这一前提不变，因为输赢可观测） | 片段 |
| PolicyGuard | arXiv 2606.12896（ICML 2026） | 测试时逐步检测，信号是高斯过程后验方差；7 个 RL 游戏，AUROC 约 0.86；无棋类 | 无 | 摘要 / 片段 |
| Provable Defense against Backdoor Policies in RL | arXiv 2211.10530 | 可证明防御 | 无 | 仅标题（来自 Plan2Cleanse 参考文献） |
| Daze：不可信模拟器下的无奖励后门 | arXiv 2602.05089 | 操纵环境动力学，而非奖励，来植入后门 | 部分（类比把自我对弈模拟器当攻击面） | 片段 |
| TRAP: Tail-aware Ranking Attack for World-Model Planning | arXiv 2605.01950 | 推理时补丁扭曲想象轨迹的排序，不改权重 | 无（对抗样本，不是后门） | 片段 |
| Targeting World Models to Compromise Robot Learning Pipelines | arXiv 2606.09499 | 只通过图像操纵世界模型，在下游策略里植入后门 | 无 | 片段 |
| Trojans in AI（TrojAI）Final Report | arXiv 2602.07152 | DRL 木马主要在 MiniGrid（LavaWorld 等）；treacherous turn 式行为"很容易"通过投毒或模仿学习植入；RL 两轮检测 AUC 0.99 / 0.95 | 部分（检测前提，见第 2 节） | 全文中 RL 相关章节 |
| Beyond Owls: Subliminal Learning Can Transfer Capabilities and Backdoors | arXiv 2610.10657（2026-10） | 在教师生成的数字序列上微调学生，"在 agentic 国际象棋环境里黑掉环境"的倾向从 10.9% 升到 58.3% | 部分（G5：蒸馏传递后门的机制；不涉及棋类引擎） | 摘要 |
| Pay Attention to the Triggers: Backdoors That Survive Distillation | arXiv 2510.18541 | LLM 后门大多传不到学生；给出能传过去的构造 | 部分（G5 机制，LLM） | 片段 |
| Anti-Distillation Backdoor Attacks | ACM MM 2021 | 图像模型后门在自然蒸馏后完整保留 | 部分（G5 机制） | 片段 |
| AF2-Mutation | arXiv 2305.08929 | 改 3 个残基即可大幅改变 AlphaFold2 的预测；属对抗样本，不是后门 | 无（G5：没找到任何 AlphaFold 后门或投毒工作） | 片段 |
| The Algorithm Is Not the Behavior（Lc0 可解释性） | arXiv 2508.21380 | Lc0 前瞻能找到正确战术，但最后一层的安全先验会把它覆盖掉；这是自然失效，不是后门 | 无（可作"自然缺陷 vs 植入缺陷"的对照） | 片段 |
| Super(ficial)-alignment | arXiv 2406.11431 | 强模型会欺骗弱监督者，严重程度随能力差增大（偏好对齐场景） | 部分（方法论相邻） | 片段 |
| 腾讯朱雀实验室 2020 XCon 披露 | 媒体报道（钛媒体、澎湃等） | 逆向模型文件、直接操纵神经元构造后门；另有"数据木马"、浮点低位藏码 | 无（与棋类无关；属权重级供应链类比） | 仅媒体报道 |
| 事实项：KataGo 官方分布式训练的种子对局通道 | Tseng 2406.12843、Wang 2211.00241 正文 | 自 2022-12 起，0.08% 的自我对弈对局从手写局面开局（用于防御环形攻击） | 部分：说明官方训练管线里已存在"人工注入局面"的入口（推测档：可作投毒入口，未见任何报道） | 全文 |

### 2. 对 round 1 说法的更正与 G4 全文升级

- **Scheurer 帖子：作者归属错误。** 帖子作者是 Bowman，正文只谈语言任务，失准靠提示或训练去误导。Scheurer 的评论提议的是噪声或 ε 混入的失准，不是触发器后门。它对我们的意义是概念层先例，只要求"弱方能否借助失准的专家"，不涉及检测或定位植入缺陷。
- **Lan 2022（2211.03769，全文 PDF）。**
  - 威胁模型是测试时加 1–2 颗"无意义子"，不是训练期投毒。
  - 搜索量只测到 50 次模拟（KataGo，ZZ 数据集）。价值攻击 2STEP 成功率：1 次 1.00，10 次 0.95，25 次 0.89，50 次 0.58。策略攻击 2STEP：25 次 1.00，50 次 0.68。50 次以上没有数据。
  - 所以"搜索压制攻击"只是极小搜索区间内的趋势。
  - 他们的评估框架本身值得注意：用更强的 examiner（KataGo 40b、800 次模拟）加更弱的 verifier（人类），要求扰动后的局面与原局面等价、且人能验证。这是"超人失效如何让弱方验证"的现成方法论，部分覆盖我们的评估设计。
- **Wang 2023（2211.00241 v4，全文 HTML，含附录 H、N）。**
  - 环形对抗者对 Latest 的胜率：4096 visits 时 97.3%，10^6 时 82%（50 局），10^7 时 72%（50 局）。
  - 结论是搜索"是有效工具，但单靠它得不到完全鲁棒"。附录 H 表 H.1/H.2：victim visits 从 1.6k 加到 500k，决定性局面里排第一的正确着法变多。
  - 不利点：对高搜索量受害者训练出的对抗者，会偏好被吃前间隔更长的局面。
  - 全文没有出现 backdoor 或 trojan，"poison" 只在比喻里出现一次。
- **TrojanTO（2506.12815，全文 HTML 前 100k / 108k 字符）。** 触发器放在状态维度（第 1–3 维，值用 MI-FGSM 优化），**不改 RTG**，原文 "the RTG will not be modified"。它是对预训练模型的训练后攻击，只用 10 条轨迹；不涉及搜索或规划。round 1 若暗示触发器放在 return-to-go 条件输入里，应改正。
- **Universal Trojan Signatures（Acharya 等，NeurIPS'23 workshop）：论文正文没取到**（OpenReview 人机验证拦截）。以下来自 TrojAI 终期报告 §5.B.1.d 的转述：
  - 检测信号是 Jacobian 或归因敏感度。
  - 前提是：被植入木马的智能体会给"通常应与负面结局相关的状态"（例如走进岩浆）赋高价值。
  - 检测器是在数百个带标签的干净 / 木马模型上训练的元分类器（RL 数据集每轮 714 / 814 个模型）。
  - 推导档：这个前提要求评估者知道哪些状态是坏的。在超人棋类场景里，这正是做不到的，所以该方法不能直接迁移。
- **Angel or Demon（2605.14587，ICML 2026，全文 HTML 前 100k / 170k 字符）。**
  - 环境只有经典控制、物理控制和 PyBullet 机器人，补充实验在 MPE；攻击方法为 TrojDRL、BadRL、SleeperNets、UNIDOOR；无棋类、无 MCTS、无自我对弈。
  - SAM 会放大后门：机器人任务训练后阶段 ASR 从 0.178 升到 0.326。Weight Clipping 和谱归一化会削弱后门。
  - 后门训练让损失地形尖锐度的范围扩大 635%，作者提议把它作为检测信号，但承认跨任务阈值难定。
  - 与我们的关系：属于一般 DRL 结论，在大型搜索引擎上是否成立未验证（类比档）。
- **Shen & Xia（2007.11820）复核。** 目标确为 SAI；背景训练数据只有 4511 局人类棋谱，规模很小，属监督式训练，不是生产级自我对弈。round 1 的结论不变，但"已在真实围棋 AI 上植入后门"要加这个规模限定。
- **Tseng 2024（2406.12843）。** 前 100k / 127k 字符里没有出现 backdoor、trojan 或 poisoning。搜索量从 1 测到 65,536 visits。

### 3. 各缺口的查询量与关键查询

- **G1 同义词轮：21 条 WebSearch（extended 模式）。** 先列词表：backdoor / trojan / neural trojan / planted vulnerability / hidden trigger / sleeper / data poisoning / model poisoning / weight poisoning / supply-chain / trojaned checkpoint / malicious fine-tuning，交叉 Go / Baduk / Weiqi / chess engine / NNUE / Leela / Lc0 / AlphaZero / MuZero / self-play / game-playing agent / superhuman / expert-level / beyond-human。关键查询：
  - "trojan neural network chess engine Leela Lc0 hidden trigger"
  - "backdoor attack AlphaZero MuZero self-play poisoning"
  - "KataGo distributed training malicious contributor poisoned self-play data"
  - "NNUE Stockfish backdoor poisoned network"
  - "Weiqi OR Baduk ... backdoor"
  - "shogi engine OR poker bot backdoor"
  - "trojaned checkpoint OR malicious fine-tuning RL game agent"
  - "model organism backdoor superhuman domain chess testbed"
  - "planted backdoor superhuman Go OR chess AI detect without ground truth weaker auditor"
  - "poisoning self-play replay buffer AlphaZero"
  - "backdoor self-play opponent pool"
  - 结果：没有一条命中"在 AlphaZero 或 Lc0 类引擎里植入触发器"。最近邻仍是 Shen & Xia。
- **G2 参考文献挖掘：5 次 Semantic Scholar 调用，另加 1 次 Tseng 全文。** 参考文献条数：Shen & Xia 6 条；Wang 84 条；Tseng 在 S2 上为 0 条，改从 arXiv HTML 取到约 70 条；Plan2Cleanse 63 条；Fluri 123 条。
  - Shen & Xia、Wang、Tseng、Fluri 的参考文献里**没有任何一条涉及棋类或搜索型智能体的后门或投毒**。
  - Wang 和 Tseng 引了 Timbers et al. 2022（AlphaZero 近似可利用性），属对抗方向，不是后门。
  - Plan2Cleanse 的参考文献标出 RL 后门相关 11 条：Fox in the Henhouse、Beyond Training-time、PolicyCleanse、Provable Defense、Temporal-Pattern Backdoor 2205.02589、MARNet、BIRD、SHINE、Neural Activation Space mitigation 2407.15168、UTS、Stop-and-Go。全部是非棋类。
- **G3 非英文：10 条。** 中文 6 条（围棋 人工智能 后门攻击；强化学习 后门 博弈 软件学报；KataGo 漏洞 绝艺 星阵；围棋 AI 数据投毒 棋谱；腾讯朱雀 AI 后门；计算机学报 / 中国科学 DRL 后门综述），日文 2 条，韩文 2 条。
  - 没找到任何中、日、韩文的围棋 AI 后门研究。命中的都是对 Wang / Tseng 的翻译和新闻，外加《计算机学报》2023 年的 DRL 对抗攻防综述（只讲对抗攻击）。
  - 社区里有星阵"套用 KataGo 旧权重"的传闻（知乎），未证实。若属实，说明权重来源问题在现实中存在（推测档）。
  - 另有 2026-08 柯洁"反 AI 战术"报道，属对抗，不是后门。
- **G4 全文升级。** 读到全文并做过定向追问的：Lan（PDF）、Wang（HTML 全文，含附录 H、N）、TrojanTO（92%）、Angel or Demon（59%）、Bowman 帖子及 Scheurer 评论、TrojAI 终期报告 RL 章节、Miller 2020（PDF）、Tseng（79%），另复核了 Shen & Xia。UTS 原文**没取到**。另做了 4 条辅助搜索。
- **G5：3 条查询**（AlphaFold 后门 / 投毒；超人引擎蒸馏投毒；Lc0 蒸馏）。
  - 没找到 AlphaFold 或其他超人窄域系统的后门或投毒工作，只有对抗样本（AF2-Mutation）。
  - 没找到"从超人引擎蒸馏时投毒"的工作。Lc0 社区确有蒸馏网络（dkappe），但没有安全分析。蒸馏能否传递后门的机制证据都在 LLM 和图像上（2510.18541、ACM MM'21、2610.10657）。

本轮共约 38 条 WebSearch、5 次 S2 API 调用、约 12 次全文抓取。按"真正取到正文并定向追问"计，本轮读了 8 篇加 1 份报告的章节。

### 4. 边界申报
- **没覆盖到的来源**：
  - 知网 / 万方正文：只经过网页搜索，没有直接检索知网，CNKI 里的中文学位论文可能漏掉。
  - 日本 CiNii / J-STAGE、韩国 KCI / RISS：没有直接检索。
  - KataGo Discord、lczero Discord、Leela Google Group 只抽到零星片段。社区内部若讨论过"恶意贡献者上传毒数据"，我们看不到。
  - 在审稿件、非 arXiv 的安全会议（USENIX Sec、CCS、S&P 2026 的正文）只经过搜索引擎间接覆盖。
- **没取到全文的**：
  - UTS 原文（OpenReview 拦截），只能用 TrojAI 报告的转述。
  - Fake Nash、Fox in the Henhouse、PolicyCleanse、PolicyGuard、Beyond Owls 都只读了摘要或片段，其中有没有棋类附录没有核实。
  - Wang 正文前 100k 字符中的数值是 WebFetch 小模型转述的；附录 H 是我自己 grep 原文核对的。
- **并行未公开工作**：已知同类组织（FAR AI 等）有持续的围棋鲁棒性工作，未公开的后门方向无法排除。
- **知识截止**：我的训练截止在 2026-06，之后的论文只靠搜索覆盖，2026-09 到 11 月的新工作可能漏检。
- **分工**：Engels 2025 和 Marks 2025 按约定由另一个代理读，本轮没碰。
- **对抗式检索**：只在 G1 里做了 2 条（"planted backdoor … weaker auditor"、"model organism … chess testbed"），不是完整的一轮。
- **轮次**：这是该想法接受的第二轮检验。按 5.2 条，"未见正面撞车"只代表搜得更多一点，不代表这个想法更强。