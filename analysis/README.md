# analysis/

纯 CPU 脚本，不加载网络、不用 GPU。依赖：Python 3 + numpy。所有脚本用 `python3 -I` 运行；读下载数据的脚本只读文本、不执行。

| 路径 | 用途 | 结果写在 |
|---|---|---|
| `theory/puct_threshold.py` | Q5：KataGo PUCT 下只污染策略网络的后门的冲淡阈值（闭式与模拟） | `docs/DERISK_ZERO_GPU_2026-11-27.zh-CN.md` §1 |
| `theory/power.py` | 检出功效：信号 d × 触发比例 × 局面数 → 模型级检出率 | 同上 §2 |
| `theory/elo_ci.py` | E2 人类锚点所需对局数 | 同上 §3 |
| `s1/criteria.py` | S1 合同的判读函数（判读归本模块） | `docs/CONTRACT_S1_DRAFT_2026-11-27.zh-CN.md` |
| `s1/rehearse.py` | S1 判据离线预演（能满足 / 不能满足 / 零假设误判率），不符即非零退出 | 同上 §4 |
| `s1/rehearse_e3_power.py` | E3 判据功效曲线（参考，不作门槛） | 同上 §4 |
| `tools/select_ladder.py` | 从 katagotraining.org 网络列表选阶梯、核 E3 分界网络 | 同上 §5 |
| `tools/list_goattack_sets.py` | 从 goattack.far.ai 前端 JS 提取对局集目录 | 同上 §5 |
| `tools/summarize_goattack_csv.py` | go_attack 对局集结果层读数 | 同上 §5 |

输入数据（`nets.tsv`、go_attack CSV）不入库，来源 URL 见各脚本与去风险报告。
