# EmbodiedBench 原生 Baseline 实验日志

本文件只记录直接调用 EmbodiedBench stock evaluator/planner 的基础模型结果，不调用
`vista_skill` CLI 或 adapter，不注入 Skill/Memory，不启用 ledger、VTCA、attribution、teacher、
patch 或 candidate gate。

## N-B1：Qwen3-VL-8B-Instruct / EB-NAV full official eval

### 目的

补齐 `context4agent/latex/sec/4_experiments.tex` 中 Qwen3-VL-8B-Instruct 的 EB-Navigation
五列基础执行数据。实验名为
`qwen3vl8b_native_official_full_20260827_r1`。

### 冻结设置

- 入口：`python -m embodiedbench.main env=eb-nav`，工作目录 `EmbodiedBench/`。
- executor：`Qwen/Qwen3-VL-8B-Instruct`，stock `EBNavigationPlanner` + stock
  `RemoteModel`，remote vLLM endpoint `http://192.168.1.185:8000/v1`。
- 五个官方 eval sets：`base`、`common_sense`、`complex_instruction`、
  `visual_appearance`、`long_horizon`；每个 60 episodes，`down_sample_ratio=1.0`，合计 300。
- 原生默认口径：temperature=0、max completion tokens=4096、n-shot=3、resolution=500、
  chat history=false、language only=false、multiview=false、detection box=false、
  multistep=false、visual ICL=false。`truncate=true` 因 chat history 关闭而不生效。
- EB-NAV stock CLI 不暴露 rollout seed；环境初态来自冻结 dataset。只有发生 JSON 解码失败时
  planner 才会使用未显式设 seed 的随机 fallback，因此必须另外报告 `planner_output_error`。
- primary metric：每个 subset 的 mean Task Success；Avg. 为五个 subset success 的宏平均。

### 命令与产物

完整 console：
`running/embodiedbench_native_nav/qwen3vl8b_native_official_full_20260827_r1/console_rerun1.log`。
Stock episode/image/config/summary 输出：
`EmbodiedBench/running/eb_nav/Qwen3-VL-8B-Instruct_qwen3vl8b_native_official_full_20260827_r1/`。

首次命令在 0 episode 前被 Hydra 拒绝，因为 stock 顶层 config 未声明 `truncate`，需要使用
`+truncate=true`；该启动失败保存在同一记录目录的 `console.log`，不计入实验。`rerun1`
只修正 Hydra 参数语法，没有改变 evaluator、planner、prompt 或模型设置。

`rerun1` 完整跑完 Base、Common Sense、Complex Instruction 后，承载进程的 Codex 执行会话
在 Visual Appearance 3/60 时随任务中断退出；console 未记录模型、HTTP 或模拟器异常。前三个
60/60 子集有效，原目录下 Visual 的 3 条部分结果不计入。Visual 与 Long 从全新 stock exp name
`qwen3vl8b_native_official_full_20260827_r1_resume1` 完整重跑，console 为
`console_resume1.log`。最终汇总器显式从两个 run roots 取对应完整子集，不拼接 partial episodes。

### 结果

五个 subset 均通过 60/60 完整性检查；最终采用的 episode 文件编号均连续覆盖 1--60，合计
300/300。独立汇总器与各 subset 的 stock `results/summary_all.json` 交叉核对一致。

| Eval set | Episodes | Successes | Task Success | Mean reward | Mean env steps | Mean planner steps | Total planner errors | Mean elapsed (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Base | 60 | 35 | 0.583333 | 0.051718 | 15.3667 | 6.6500 | 1 | 32.6799 |
| Common Sense | 60 | 32 | 0.533333 | 0.048818 | 15.5167 | 6.9000 | 0 | 39.1702 |
| Complex Instruction | 60 | 30 | 0.500000 | 0.044248 | 15.9500 | 6.7333 | 0 | 33.4246 |
| Visual Appearance | 60 | 29 | 0.483333 | 0.044992 | 15.9500 | 6.5000 | 1 | 32.4181 |
| Long Horizon | 60 | 21 | 0.350000 | 0.025464 | 17.9833 | 7.7667 | 1 | 38.3116 |

五个 Task Success 的宏平均为 **0.490000**。LaTeX 两位小数口径下，EB-NAV 的填表顺序
`Avg. / Base / Com. / Comp. / Vis. / Long` 为：
`0.490 / 0.58 / 0.53 / 0.50 / 0.48 / 0.35`。

机器可读汇总（包含 config、dataset 和 300 个 episode result 的 SHA-256）为
`running/embodiedbench_native_nav/qwen3vl8b_native_official_full_20260827_r1/analysis.json`。
其 SHA-256 为 `ee2ca604058741830f788c2cef5a6e95f64159f928a19b7001a2683ba517a1ef`；
`console_rerun1.log` 与 `console_resume1.log` 的 SHA-256 分别为
`993f9df746cbf6b51b0cc435901ba1bdd2e235358fbbeaed6908c5b9b364a543` 和
`77ea30aa03fc2153a84b2df3266f4c95f19b584336ffe52a545e7c1b5c2c2325`。
该文件的 `claim_scope` 固定为 stock EmbodiedBench baseline；禁止用此前 VISTA adapter 的
20-episode diagnostic 代替本结果。

需要特别注意：本次 Long Horizon 的 0.35 与主表中历史 `No Skill/Memory` 行的 0.02 差异很大，
Complex Instruction 也由历史行的 0.62 变为 0.50。由于历史行当前缺少与本次同等级的原始
run provenance，不能把它当作同口径复现值；本次新填的 `Ours / Qwen3-VL-8B-Instruct`
EB-NAV 空缺只采用上述冻结 stock protocol 的 300-episode 结果。
