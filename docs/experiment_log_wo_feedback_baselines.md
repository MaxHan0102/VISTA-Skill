# W/o feedback baseline 最终实验日志（2026-09-10）

## 目的与最终状态

检验 VLM executor 在移除仿真器语义反馈后、保留 RGB 和动作历史时的任务完成能力，
为 VISTA-Skill 的 w/ 与 w/o feedback 双轨对照建立无 Skill/Memory 基线。
四个模型各运行 EB-Habitat 和 EB-Navigation，共八组 **2,400 个完整任务**，均已完成。
这些结果不是 VISTA-Skill 方法结果，也不能用成功率筛选或丢弃正常结束的失败任务。

## 实验协议

- 入口：`evaluate_closed_loop_feedback.py`；恢复：`resume_closed_source_wo_feedback.py`。
  Qwen 入口为 `run_qwen_feedback_experiment.sh` / `run_qwen_wo_feedback.py`。
- 使用 stock evaluator/planner/RemoteModel 和原任务顺序、动作预算、多动作计划、解析与评分。
  不修改 EmbodiedBench 官方源码，不注入 Skill、Memory、teacher 或演化。
- 单帧 RGB、500 分辨率、seed 0；Habitat 10-shot，每个subset 50任务；NAV 3-shot，每个subset 60任务。
  GPT-5 按官方路径不传 temperature；其他模型 temperature=0；最大生成token为4096。
  seed 是进程/环境设置，不代表为所有模型API显式传入seed，也不保证断点恢复逐像素一致。
- `rgb_only`：Habitat 使用官方 `use_feedback=False`；NAV 删除两处历史反馈插值；
  planner历史只保留动作，不传真实语义结果；禁用隐藏失败标志触发的提前重规划。
  RGB、指令、公共动作历史仍可见，环境终止与真实评分信息保留，原始info仅记录/评分。
- 工程适配：OpenLux或Qwen endpoint、结构化schema兼容、审计与有限API/进程重试。
  GPT字典schema补充对象 `additionalProperties:false`；两个Gemini展开 `$defs`/`$ref`，
  仍用原Pydantic字段和SDK解析。各任务schema来源版本保存在索引中，见下方机器可读记录。
- 已有完整 `episode_N_final_res.json` 跳过，含任务失败；中断任务从头恢复，残留归档。
  新任务结果直接补入主目录；进程日志/API审计分 `_sessions` 保存，历史 `_resume` 保留。
  旧结果整合时保留第一份完整记录和哈希，不以稍后的成功结果替换失败结果。

## 最终成功率

Avg. 使用未舍入分项成功率的宏平均；各环境类别等量，因此也等于整体成功数/300。
列顺序匹配论文主表，Habitat 的 Vis. 位于 Spa. 之前。下表单位均为百分比。

### EB-Habitat

| 模型 | 成功数 | Avg. | Base | Com. | Comp. | Vis. | Spa. | Long |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gpt-5.4-mini | 59/300 | 19.67 | 58.00 | 4.00 | 16.00 | 14.00 | 20.00 | 6.00 |
| gemini-3-flash-preview | 74/300 | 24.67 | 60.00 | 12.00 | 24.00 | 24.00 | 26.00 | 2.00 |
| gemini-3.5-flash | 160/300 | 53.33 | 82.00 | 46.00 | 48.00 | 62.00 | 34.00 | 48.00 |
| Qwen3-VL-8B-Instruct | 65/300 | 21.67 | 50.00 | 6.00 | 16.00 | 24.00 | 26.00 | 8.00 |

### EB-NAV

| 模型 | 成功数 | Avg. | Base | Com. | Comp. | Vis. | Long |
|---|---:|---:|---:|---:|---:|---:|---:|
| gpt-5.4-mini | 136/300 | 45.33 | 58.33 | 53.33 | 50.00 | 55.00 | 10.00 |
| gemini-3-flash-preview | 166/300 | 55.33 | 56.67 | 68.33 | 65.00 | 60.00 | 26.67 |
| gemini-3.5-flash | 193/300 | 64.33 | 71.67 | 70.00 | 70.00 | 58.33 | 51.67 |
| Qwen3-VL-8B-Instruct | 123/300 | 41.00 | 48.33 | 41.67 | 56.67 | 46.67 | 11.67 |

## 中断、重试与成本解释

已出现的主要问题：OpenLux schema HTTP 400（已适配）、连接失败、输出长度截断，
以及 NAV 在恢复 FloorPlan228 时的模拟器超时。完成任务保留，原预算不因错误而提升。
Gemini 3.5 Habitat 的 Spatial 子集反复出现 `LengthFinishReasonError`，明显拖慢运行；
官方解析重试允许同一状态再次请求，因此需要同时报告可靠性和成本，不能只看成功率。
这不是通过重跑完整任务挑选成功结果，但无限解析重试属于当前协议的实际限制。

下表统计原始目录及全部 `_resume` / `_sessions` 审计，排除pilot，包含中断尝试；
SDK内部重试不逐次记录，早期错误日志可能缺失，不能将这些计数当作完整计费。

| 模型/环境 | 请求记录 | 成功响应记录 | 记录的错误类型与次数 |
|---|---:|---:|---|
| gpt-5.4-mini / eb-hab | 1712 | 1650 | BadRequestError: 45, APIConnectionError: 16 |
| gpt-5.4-mini / eb-nav | 1230 | 1230 | 无记录 |
| gemini-3-flash-preview / eb-hab | 732 | 689 | BadRequestError: 12, LengthFinishReasonError: 31 |
| gemini-3-flash-preview / eb-nav | 1438 | 1278 | LengthFinishReasonError: 160 |
| gemini-3.5-flash / eb-hab | 1819 | 1364 | BadRequestError: 17, LengthFinishReasonError: 438 |
| gemini-3.5-flash / eb-nav | 1345 | 1344 | LengthFinishReasonError: 1 |
| Qwen3-VL-8B-Instruct / eb-hab | 1084 | 1084 | 无记录 |
| Qwen3-VL-8B-Instruct / eb-nav | 1426 | 1426 | 无记录 |

## 服务器和路径

闭源模型使用 `https://api.openlux.ai/v1`，其上游版本/路由不由本仓库控制。
Qwen HAB 使用 `http://192.168.1.185:8000/v1`，NAV 使用 `http://192.168.1.173:8001/v1`，
均公布 `Qwen/Qwen3-VL-8B-Instruct`、16384上下文；服务器GPU/TP/权重量化未独立验证。
客户端旧IPv4网关不可达，经DHCP临时地址及两条目标路由恢复，默认路由和IPv6保留。
网络记录：`running/test/qwen_feedback_network_20260910/network_recovery.json`。

八组主目录均位于 `running/closed_feedback/<模型>_<环境>_v2_full/`。
`resume_summary.json` 是全组汇总，`resume_state.json` / `episode_provenance.json` 给出每个
被接受结果的位置、哈希、任务身份和schema版本。类别原生汇总位于 `subset/results/`。
[机器可读最终结果](wo_feedback_baseline_results_20260910.json) 保存精确分项成功数、
配置/索引哈希、schema版本分布和可观测API计数；token仅含已记录的成功响应用量，
不包含失败请求可能产生的消耗，不能视为总计费成本。

## 双轨叙事与后续方法验证

主表非灰色行为w/ feedback，灰色行为本次w/o结果。现有数字支持不同模型和任务对
仿真器语义反馈有不同程度的依赖：部分明显下降，部分变化较小。此处“特权信息”特指
仿真器直接提供的语义动作结果/失败原因，不包括RGB这种正常观测。
但旧w/结果与本次的endpoint、schema、重试和模型上游版本尚未逐项确认完全一致，
不能将全部分数差异直接当成严格控制的反馈因果效应，亦未作统计显著性声明。

后续优先在EB-Habitat以同一Qwen executor、任务和预算验证VISTA-Skill相对无Skill基线及
受控Skill基线的增益，同时报告更新可靠性与teacher成本；需要严格反馈效应比较时，
补跑当前脚本的匹配full轨。Qwen本次数据应放无Skill Qwen灰色行，
不能填写VISTA-Skill方法行。服务器端LaTeX仅参考，本次未修改。

## 验证与复现

八组通过完整任务计数、数据集/结果哈希和任务身份检查；各300，无缺失或部分结果。
相关单元/SDK mock测试33项通过；真实Habitat与NAV均完成pilot到全量验证。
详细启动、适配与历史恢复记录见 [运行说明](closed_source_feedback_baselines.md)。

提交前完整回归：`python -m pytest -o addopts='' -q`，**335 passed**（2026-09-10）。
