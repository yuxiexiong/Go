# 第三轮查重：主线整体（2026-11-27）

对象：`docs/PROPOSAL_v1_2026-11-27.zh-CN.md` 的主线作为一个整体。此前两轮查的是子主张 C1–C4，主线的审计窗口、参照选择、可检性拆分、痛点表述都没查过。

检验量：12 条网页检索；Semantic Scholar 被引列表 4 篇取到（Engels 2504.18530 共 28 篇、Wang 2211.00241 共 39 篇、Tseng 2406.12843 共 4 篇、Fluri 2306.09983 共 64 篇）。Control Tax 2506.05296 两次被限流，没取到；Shihab 2606.00414、MIST 2605.21146 暂无被引；2602.16984 有 3 篇。**读到的都是摘要和检索片段，正文 0 篇。**

第 4 步词表（先列后搜，只检索了一部分词）：
- 监督方：scalable oversight、weak-to-strong、trusted/untrusted monitoring、auditor、overseer、judge、verifier
- 强度：superhuman、beyond human-level、capability gap、Elo gap
- 缺陷：backdoor、trojan、sleeper agent、planted flaw、hidden objective、model organism、sabotage
- 方法：model diffing、crosscoder、checkpoint diffing、differential testing
- 时机：evaluation timing、capability threshold、pre-deployment audit
- 场地：KataGo、Leela、AlphaZero、chess engine

## 1. 事实约束（坏消息先说；只放会改变做法的，不放"有人先做过"）

1. **参照检查点审计受漂移约束。** MIST 2605.21146 报告：被审模型离干净参照越远，检出越差。这是 Q4 和 G0 地板的直接旁证。（类比档）
2. **"能力认证测不到后门"作为认识已有。** 2602.16984 证明触发在评测中暴露量为 O(1) 时，黑盒评测估不准部署风险；Draguns 2406.02619 构造了引不出的后门；Casper 2401.14446 论证黑盒访问不足。本课题只能写成超人宿主上的定量（Q6）。
3. **AUC 会掩盖低误报率处的失败。** LessWrong "monitor identity matters more than tier"（单一设置的博客分析）与 Kale 2508.19461 都用固定误报率下的 TPR。主读数按此选。

## 2. 三列表

| 部件 | 先例支持（资源） | 事实约束与竞争解释 | 谁会改做法 | 判定 |
|---|---|---|---|---|
| Q1 真超人宿主上检出率随强度差的曲线 | Engels（拟合流程）、Control Tax（AUC 协议）、Kale、Marks、Fluri（代码公开） | Engels 400 Elo 以上是模型外推（检索确认无实测）；"更强模型更会藏后门"在对抗式检索里没找到直接测量，只找到"攻击者训练强度改变可检性"（2409.00399） | 用缩放律排审计预算的人；控制评测设计者 | **A**：空着 |
| Q2 可用性 × 隐蔽度拆分 | Engels F.3 与 Control Tax C.3 的口径差 | 本轮没有专门检索 | 控制评测设计者 | 待查 |
| Q3 审计窗口 | AuditBench 2602.22755（隐藏行为熬过对抗训练）；Anthropic 破坏评测 2410.21514 按能力门槛设检验 | LessWrong "defeating introspection adapters" 指出审计论文很少讲清审计时机，无检验 | 分级安全框架 | **A**：空着 |
| Q4 参照选择 | MIST、Diff-SAE 2605.07324、Delta-Crosscoder 2603.04426（ICLR 2026）、PoTS 2510.15106、RFTC | 方法层已占；漂移越远越差 | 拿历史版本当参照的审计方 | 方法层 B；改为"跨越人类水平的谱系上的漂移–可检权衡 + 参照选择处方"后为 **A** |
| Q5 搜索 | 第二轮已查 | 证据双向 | 依赖推理时搜索的系统 | 维持 **A** |
| Q6 能力认证与后门 | 2602.16984、Draguns、Casper | 理论与构造已有 | — | 作为认识是 **B**；作为定量子结论保留 |
| 竞争方案 | Bootstrapped Monitoring 2606.11998（LLM，中间层监控者审更强的智能体）；2608.16190（小监控者对强攻击者） | 前者依赖可读思维链 | — | 讨论，不设臂 |

## 3. 第 5 步引用链里值得留意的

- Engels 的 28 篇被引：没有超人宿主上的实验。相关的有 Mahran 2610.30803（评估与监督极限的划分理论，第二轮已登记）、Bootstrapped Monitoring 2606.11998、2605.28807（可扩展监督的保守度校准）。
- Wang 的 39 篇被引：没有"超人博弈宿主 + 植入后门 + 审计"。可用的有 "Analysing KataGo: A Comparative Evaluation Against Perfect Play"（2024，精确解对照，可能即 Husna & Müller）、Casper 2401.14446。
- Tseng 的 4 篇被引、Fluri 的 64 篇被引：没有相关撞车。
- FAR.AI：检索索引里 2026 年没有相关条目；未公开工作查不到。

## 4. 边界

见 PROPOSAL §10。

本轮 12 条检索的归属（有一条兼两步）：
- 第 1 步（抽象阶梯）：超人区间的监督实测；审计时机与能力门槛（兼第 3 步）。
- 第 2 步（方法）：模型差分与 crosscoder；以早期检查点为参照的检测。
- 第 3 步（蕴含）：能力评测排除不了后门；可信监控与强度差。
- 第 4 步（同义词）：weak-to-strong 与后门检测。
- 第 6 步（对抗）：超人博弈 AI 上的植入缺陷审计；FAR.AI 2026；监督缩放律外推是否被检验；"更强模型更会藏后门"。
- 核对：Bootstrapped Monitoring 原文信息。

各步密度都低（每步 1–4 条），按准则 5.2 只能算搜得少。
