# S0 撞车排查补完（2026-11-27）

补 PROPOSAL v1 §10 列出的未覆盖项：FAR.AI 近两年产出、Control Tax 被引、Q2 拆分专门检索、方法代码是否公开、对抗式检索。由子代理执行，我按青蒿素原则重新分类后写成本文。

检验量：WebSearch 14 条（Q2 7 条，对抗式 7 条，有重叠）；WebFetch 约 14 页；Semantic Scholar API 约 8 次（摘要请求被限流，部分改用 arXiv 页面）；OpenAlex 2 次；arXiv API 4 次。**读了正文的只有 Control Tax 和 LinuxArena，各读前 10 万字符**，其余为摘要或检索片段。

## 0. 结论

**没有找到撞车。** 没有人在超人棋类 AI 上植入后门再做审计实验，也没有人在超人区间实测过监督缩放律。FAR.AI 2025–2026 年没有围棋或 KataGo 的新工作（只读了题目）。

## 1. 事实约束与竞争解释（会改变做法的）

1. **Q2 的拆分已有定义来源，而且有一个初步答案。**
   - LinuxArena 2604.15384 §5.1（读了正文）：successful sabotage = P(undetected ∧ side task successful)，evasion rate = P(undetected | side task successful)，后者"isolates the stealth capability from the ability to complete the side task"。§5.2 说最弱的攻击者破坏率低是 "driven partly by lower side-task completion and partly by lower evasion conditional on success"；§4.2 说最强攻击者的低破坏率 "primarily reflect the cost of stealth rather than raw side-task feasibility"。
   - Control Tax 2506.05296（读了正文前段）：监控 AUC 只在 "successfully backdoored and validated" 的样本上算。附录 C.3："once we filter the backdoors on passing the public tests, ... the difference in attackers disappears"。
   - 影响：Q2 改为引这两篇作定义来源，主张落在"在连续强度轴、超人区间上测这个分解"。Control Tax C.3 是 H-平 的直接依据，冻结预测时写入。
   - 两篇都没有把分解做成随攻击者强度连续变化的曲线，LinuxArena 原文承认没做这个分析，攻击者强度只有几个离散模型。
2. **"越过人类水平"要用对局条件定义。** 2026-07 申真谞在让两子的条件下 2–1 赢过 KataGo，KataGo 配置记录很差（据称每步约 20 秒、不 pondering、网络未披露）。来源：LessWrong 帖子（读了正文，引 Korea Times）。不否定远超人类，但越线位置必须写成"网络 × 搜索量/思考时间"，不能只写检查点编号。（推导档）
3. **Q4 的竞争解释：后门模仿弱检查点。** Auditing Games for Sandbagging 2512.07810（检索片段）：被训练成模仿弱模型的模型能躲过检测。若被植入的强检查点在触发局面上的走法分布像某个历史检查点，参照差分会失效。影响：Q4 加一臂"触发行为模仿较弱检查点"。（类比档）
4. **参照跨谱系会掉分。** 2607.06596（摘要）：监控跨谱系迁移，1% 预算下 19% 对 41%。对 Q4 选参照有直接含义。（类比档）
5. **后门藏在哪比监控多强更重要。** ResearchArena 2607.19321（片段）：成功的后训练后门 AUC 只有 0.53–0.71，检出更多取决于后门位置。这是宿主强度之外的竞争变量，设计里要固定后门位置类型或单列。（类比档）
6. **审计者可能认的是植入方式而不是后门。** 2610.41713（摘要）：后门基准会通过代码风格泄露标签。对应到围棋：干净孪生必须和被植入臂走完全相同的微调流程（已在设计里），并要检查审计者是否只认出"被微调过"。

## 2. 三列表

| 发现 | 先例支持（资源/基线） | 事实约束与竞争解释 | 谁会因此改做法 |
|---|---|---|---|
| LinuxArena 2604.15384、Control Tax C.3 | Q2 的定义与"只在成功后门上算 AUC"的做法，直接沿用 | 初步答案：攻击者变强抬的是可用率、不是隐蔽度（LLM、离散攻击者） | 控制评测设计者：超人区间是否仍成立 |
| 2605.14283（Kim, Fang, Sandholm，摘要） | 给国际象棋引擎植入水印并统计检出，报告检出度与棋力的权衡。最近的先例，当基线 | 没有围棋；没变引擎或检出者的强度 | 给超人引擎做隐藏信号检出的人 |
| 2610.41713（摘要） | 攻击者代际差的数据点：Sonnet 5 监控抓旧模型后门 77%，抓自己的 37% | 基准会泄露标签 | 同上 |
| 2601.21112（摘要） | 监控拿到的信息多少影响检出，"less-is-more"，可对照五档审计者 | — | 设计审计者信息分档的人 |
| 2607.06596（摘要） | — | 跨谱系迁移掉分 | Q4 参照选择 |
| Auditing Games for Sandbagging 2512.07810 | 审计游戏协议 | 模仿弱模型可躲检测 → Q4 加臂 | 同上 |
| Tony Wang 等 2510.05092 Learning to Interpret Weight Differences（只看题目） | Q4 白盒差分的候选方法 | 未读 | — |
| Scaling Trends for Lie Detector Oversight 2607.01567、FIBS 探针 2610.12445（只看题目） | 相邻的监督缩放工作 | 未读 | — |

## 3. 代码是否公开

| 方法 | 仓库 | 状态 |
|---|---|---|
| Fluri 2306.09983 | github.com/ethz-spylab/superhuman-ai-consistency | 可访问；只有国际象棋，见 `docs/S0_CODE_REUSE_2026-11-27.zh-CN.md` |
| go_attack | github.com/AlignmentResearch/go_attack | 可访问 |
| Diff-SAE 2605.07324 | github.com/techsachinkr/diff-sae-backdoor-detection | 可访问（正文给出） |
| Control Tax | github.com/MikhailTerekhov/control-tax-code | 正文给出 |
| MIST 2605.21146 | 正文给的 github.com/PasiniSamuele/MIST 返回 404 | 要自己复现；作者名下 spt-trojan-detection 是否即 MIST 未核 |
| Delta-Crosscoder 2603.04426 | 没找到 | 要自己复现 |

## 4. 边界

- Control Tax、LinuxArena 只读了前 10 万字符；其余都是摘要或片段。
- Semantic Scholar/OpenAlex 被引列表可能不全，没访问 Google Scholar。Control Tax 被引：S2 12 条、OpenAlex 16 条。
- FAR.AI 只看了 research 页和 blog 页，没逐个翻子链接；Tony Wang 只按 "Tony T. Wang" 检索。
- 在审稿件、非 arXiv、非英文、未公开的并行工作没覆盖；2026 年中以后受索引延迟影响。
