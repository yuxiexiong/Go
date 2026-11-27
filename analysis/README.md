# analysis/

`theory/`、`s1/`、`tools/` 是纯 CPU 脚本，不加载网络、不用 GPU，依赖 Python 3 + numpy。`oldnet/` 需要 torch 与 KataGo python 代码。所有脚本用 `python3 -I` 运行；读下载数据的脚本只读文本、不执行。

| 路径 | 用途 | 结果写在 |
|---|---|---|
| `theory/puct_threshold.py` | Q5：KataGo PUCT 下只污染策略网络的后门的冲淡阈值（闭式与模拟） | `docs/DERISK_ZERO_GPU_2026-11-27.zh-CN.md` §1 |
| `theory/power.py` | 检出功效（只有计数检验；已被 `power_tests.py` 取代，保留作记录） | 同上 §2（已修订） |
| `theory/power_tests.py` | 检出功效：计数（三种阈值）/ 最大分数 / HC / 似然比 × (d, 暴露, M) | `docs/SELFCHECK_2026-11-27.zh-CN.md` #3 |
| `theory/puct_threshold_lcb.py` | Q5 修订：加入 KataGo LCB 选着与 FPU | `docs/SELFCHECK_2026-11-27.zh-CN.md` #31 |
| `theory/elo_ci.py` | E2 人类锚点所需对局数 | 同上 §3 |
| `s1/criteria.py` | S1 合同的判读函数（判读归本模块） | `docs/CONTRACT_S1_DRAFT_2026-11-27.zh-CN.md` |
| `s1/rehearse.py` | S1 合同 v2 判据离线预演（能满足 / 不能满足 / 零假设误判率 / 对照匹配演示），不符即非零退出 | S1 合同 v2 |
| `tools/select_ladder.py` | 从 katagotraining.org 网络列表选阶梯、核 E3 分界网络 | 同上 §5 |
| `tools/list_goattack_sets.py` | 从 goattack.far.ai 前端 JS 提取对局集目录 | 同上 §5 |
| `tools/summarize_goattack_csv.py` | go_attack 对局集结果层读数 | 同上 §5 |

| `oldnet/scripts/` | 旧网络（.bin.gz/.txt.gz、TF 变量）转 PyTorch、与 C++ 引擎比对的脚本（由子代理在服务器写成，需 torch 与 KataGo python；`run_with_oldnet_patch.py` 用猴子补丁处理 5×5 初始卷积） | `docs/OLDNET_CONVERSION_2026-11-27.zh-CN.md` |

输入数据（`nets.tsv`、go_attack CSV）不入库，来源 URL 见各脚本与去风险报告。
