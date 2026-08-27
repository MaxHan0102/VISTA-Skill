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

运行中；完成后在此写入五个 60/60 完整性检查、Task Success、planner error、步数和墙钟时间，
再更新 LaTeX 表格。禁止用此前 VISTA adapter 的 20-episode diagnostic 代替本结果。
