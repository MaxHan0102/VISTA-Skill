# 闭源模型反馈对照：优先沿用 EmbodiedBench 官方协议

最终八组结果（四模型 × 两环境，2,400任务）见
[w/o feedback baseline 实验日志](experiment_log_wo_feedback_baselines.md) 和
[机器可读结果](wo_feedback_baseline_results_20260910.json)。后文保留实现及验证的时间线。

入口：`scripts/evaluate_closed_loop_feedback.py`。当前协议是 **`stock_feedback_v2`**。
该脚本直接调用本项目内的官方 `evaluate_main()`，并使用官方 planner 和
`RemoteModel`，不再实现独立 rollout 循环。EmbodiedBench 源文件和 LaTeX 均不修改。

## 沿用的官方设置

配置从 `EmbodiedBench/embodiedbench/configs/eb-hab.yaml` 或 `eb-nav.yaml` 读取。
默认单帧 RGB、Habitat 10-shot、Navigation 3-shot、500 分辨率；保留官方系统提示、
静态示例、多动作计划、JSON schema 字段、JSON 解析、空计划/非法输出处理、执行预算、
成功/失败评分和原有日志格式。GPT/Gemini 分别调用官方 RemoteModel 的
`chat.completions.create` / `beta.chat.completions.parse` 路径；字典 schema 仅补齐
对象的 `additionalProperties: false`（见下文），其余请求参数保留。
默认 RemoteModel token 上限 4096；温度全局为 0，但官方 GPT-5 分支不发送 temperature。

任务数量由 `--episodes` 限制，完整测试用 `--episodes 0`。
Habitat 保留官方 episode iterator 和 start-index 行为，不重排数据集。
Navigation 使用原环境已有的 `selected_indexes` 选择任务。
`--seed` 在每个类别开始时设置进程和环境随机种子，不再逐 episode 重设。
官方 CLI 本身未提供同样的种子设置，所以复现历史数字时需核对其随机性约定。
NAV 仍有本项目已记录的 stock seed API 限制，不能承诺严格确定性。

## 只为需求做的适配

| 项目 | `full` | `rgb_only`（默认） |
|---|---|---|
| 官方 planner / 多动作执行 | 原样使用 | 沿用原版，不强制每步重规划 |
| Habitat 提示中的反馈 | 官方开启 | 使用官方 `use_feedback=False` |
| Navigation 提示中的反馈 | 原样使用 | 原类没有开关，运行时只删两处历史反馈插值 |
| planner 历史写入 | 原样使用 | `update_info` 只接收 action ID 和空反馈 |
| 隐藏动作失败触发的计划中断 | 原版行为 | 仅禁用该判断，继续执行既定计划直到结束或环境 done |
| 原始 info、终止、评分、落盘 | 原样使用 | 原样使用，真实成功/失败标志不改写 |

最后一项控制流适配是严格 RGB-only 所需：只关文本仍会从“失败后立刻重规划”
获得失败信号。Habitat 仅将 `if done or info['last_action_success'] == 0` 改为
`if done`；NAV 仅禁用相应失败分支。修改仅作用于本进程内绑定的方法，
通过精确匹配官方源码生成；若官方源码变化导致匹配失败则停止，避免误改。
`full` 的 `evaluate()` 完全不替换。确切替换内容记在 `config.json`。

仍由官方环境决定成功终止、无效动作上限等，这是 benchmark 的终止协议，
不能声称验证了真实机器人自主完成判断。本脚本不涉及 VISTA 的完整演化链路。

额外的工程支持包括 OpenLux 地址/密钥和 schema 兼容、输出目录和任务范围选择、种子、
终端日志与 API 审计记录，以及续跑入口的有限重试。401/403/404/422 和明确的
invalid_json_schema 立即终止；其他 400 的有限重试见续跑说明。不会自动换模型或
降级为无结构输出；未启用适配层重试的其他异常仍沿用官方 retry 行为。

## 环境与 API KEY

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate max_embench
cd /root/max/VISTA-Skill/EmbodiedBench

read -rsp 'OpenLux API key: ' OPENAI_API_KEY
export OPENAI_API_KEY
export GEMINI_API_KEY="$OPENAI_API_KEY"
```

两个系列均默认请求 `https://api.openlux.ai/v1`。上述示例共用一枚 OpenLux key；
如各有不同 key，分别设置。无需每次输入 URL。临时覆盖地址用 `--base-url`，
覆盖变量名用 `--api-key-env`。模型 ID 必须是网关实际支持的名称，脚本不替换模型。
脚本通过官方 RemoteModel 导入依赖；使用已配置的 conda 环境即可。

## 启动测试

已在终端导入两个 key 和 DISPLAY 后，一键顺序运行三个模型 × 两个环境的
w/o feedback（每组先 2 个任务 pilot，再全量）：

```bash
bash /root/max/VISTA-Skill/scripts/run_closed_source_wo_feedback.sh
```

脚本自动切换对应 conda 环境，复用当前导出的 key 和 DISPLAY，不会重新询问。
任一阶段以非零状态退出就停止整个批次；pilot 只要求正常完成，不按成功率筛选。
加 `--dry-run` 可验证全部 12 个阶段配置，不调用 API、不启动仿真、不写运行结果。
输出路径沿用 `running/closed_feedback/{model}_{env}_v2_{pilot,full}`，冲突时由
Python 入口自动添加时间戳。批处理现已传入 `--resume-existing`：固定目录已有配置时
检查配置与任务范围，跳过已有完整结果，只补缺失任务；不再重复生成整个批次。

### 中断后续跑（使用新的 resume 脚本）

已经有部分全量结果时，推荐直接继续全量组（无需经过 pilot）：

```bash
bash /root/max/VISTA-Skill/scripts/resume_closed_source_wo_feedback.sh
```

复用已导出的两个 API key 和 DISPLAY。默认扫描六组固定名字的 `*_v2_full` 目录，
跳过完成的任务，接着启动尚未开始的模型/环境；不重跑 pilot，也不自动采用单独的
debug/pilot 结果。若此前使用带时间戳的其他目录，用 `--run-dir` 明确指定。

```bash
# 只检查进度，不调用 API、不写文件
bash /root/max/VISTA-Skill/scripts/resume_closed_source_wo_feedback.sh --dry-run

# 仅续跑这一组
bash /root/max/VISTA-Skill/scripts/resume_closed_source_wo_feedback.sh \
  --run-dir /root/max/VISTA-Skill/running/closed_feedback/gpt-5.4-mini_eb-hab_v2_full
```

恢复边界是**完整任务**，不是恢复到中断动作。第一次完整落盘的官方结果优先，
包括 task_success=0；已完成的失败任务不会重跑。中断任务从开头重新执行。
每次启动按 `<类别>/results/episode_N_final_res.json` 文件名扫描，并检查 JSON 是否完整、
任务身份及已接受文件的哈希。完整结果存在就跳过，不要求 task_success=1。
缺失/损坏结果对应的残留图片、轨迹和损坏结果先移到 `_interrupted/run_时间戳/`，
记录原路径和哈希，再从头执行该任务。

**新结果直接写回原目录的类别目录**：`<类别>/results/`、逐步 JSON 和 `images/`，
均保持官方命名。每个 worker 启动前只领取一段连续缺失任务；重置前再检查结果不存在，
结果文件使用独占创建模式，避免覆盖已有结果。正常结束的失败任务不会重跑。
每次进程的配置、终端日志和 API 审计保存在 `_sessions/run_yyyymmddhhmm.../`，
调用编号在各 session 内解释；这里不保存另一份任务结果。
官方类别汇总按主目录全部最终结果重算，明确排除之前的 summary 文件；原版配置文件保留。

- `resume_state.json`：已接受结果的路径、哈希和任务身份，重启后继续使用；
- `resume_summary.json`：整个实验合并后的完成数、分组成功率、完成后的类别均值；
- `resume_expected_episodes.json`：重启任务时在调用模型前核对已知 episode ID。
- `episode_provenance.json`：主目录任务结果对应的原始运行/session、schema 版本和哈希。

原目录旧 `failure.json`/`summary.json` 保留，可能仍反映原始那次中断；
**续跑后的整体状态与填表汇总以 `resume_summary.json` 为准**。
每条被合并的任务都有唯一来源；不能再把原目录和 attempt 汇总直接相加。
成本统计应包含所有原始、旧 `_resume`、新 `_sessions` 的 API 审计，包括被放弃的中断
任务调用，不能只算最终结果。旧 `_resume` 保留作为审计来源，不再存入新的续跑任务结果。

重试有两层：默认在同一进程内对原请求额外重试 3 次（5/10/20 秒等待），
不改变模型、schema、参数或当前计划；仍失败时，控制器保存已完成任务，等待 30 秒，
再从未完成任务重启，默认额外最多 3 次。支持连接/超时、408/409/429/5xx，以及
本次出现过的原因未明确的 400；401/403/404/422 和明确的无效 schema 错误立即停止。
400 并不必然是网络问题，
持续失败会达到上限后停止，应查看实际 provider 错误。
可用 `--api-retries`、`--max-retries`、`--retry-delay` 调整；不会无限重连。

检查官方源码哈希、数据集哈希、实验配置和已接受结果哈希，发生不一致时停止，
不静默混合协议。进程锁阻止两个续跑控制器同时写同一组；原批处理的已有目录续跑也使用此锁。
Ctrl+C 会停止当前 worker 并保存已落盘进度，下次运行同一 resume 命令即可继续。

重启会重新初始化仿真器；保留官方 start-index 和 seed 行为并核对已知任务 ID，
但这不是保存/恢复完整 simulator RNG 和物理状态，因此不承诺与未中断轨迹逐像素相同。
这些重启需作为实验运行记录保留，不能声称完成了仿真状态的精确续接。

你之前的命令可以直接继续使用，现在会走官方流程：

```bash
python ../scripts/evaluate_closed_loop_feedback.py \
  --provider openai --model gpt-5.4-mini \
  --env eb-hab --track rgb_only \
  --eval-sets base --episodes 2 --seed 0 \
  --output ../running/closed_feedback/gpt54mini_hab_rgb_pilot
```

输出目录已有内容时自动追加北京时间 `yyyymmddhhmm`；同分钟冲突再追加序号。
单独调用默认创建独立运行；加 `--resume-existing` 则核对原配置并补齐缺失任务。
批处理已默认加此选项。旧结果不会覆盖，启动时打印实际目录。

Gemini 将模型参数替换为：

```bash
--provider gemini --model gemini-3-flash-preview
# 或
--provider gemini --model gemini-3.5-flash
```

模型 ID 和可访问版本需与网关及原实验核对。
`--dry-run` 只展示配置；`--smoke-policy --max-steps 4 --episodes 1` 使用固定
两动作计划验证真实仿真与日志，不读取 key、不调用 API，结果不能填主表。

全量配对（默认单帧，不加 smoke 的步数限制）：

```bash
for track in rgb_only full; do
  python ../scripts/evaluate_closed_loop_feedback.py \
    --provider openai --model gpt-5.4-mini --env eb-hab \
    --track "$track" --eval-sets all --episodes 0 --seed 0 \
    --output "../running/closed_feedback/gpt54mini_hab_${track}_official" || break
done
```

Navigation 激活 `max_embench_nav`，改为 `--env eb-nav`。需要可用的 X server：
已有 `:0` 时 `export DISPLAY=:0`；若需新启动，在单独 tmux 窗口按官方命令
`python -m embodiedbench.envs.eb_alfred.scripts.startx 1`，评测终端用 `DISPLAY=:1`。
新终端需重新导入 key。

显式 `--frames 3` 才启用官方时序图像选项，这是额外消融；两轨应同设。
默认测试优先官方单帧，论文中三帧条件需另行注明。
`--n-shots`、`--resolution`、`--max-tokens`、`--temperature` 仅在显式传入时覆盖，
其中 GPT-5 仍遵循官方不传 temperature 的分支。
此前独立客户端的 `--reasoning-effort`、`--timeout` 不再支持，避免改写官方请求和重试配置。

## 日志与结果：原版为主，审计为辅

- 终端恢复官方 Instruction、Planner Output Action、Executed action、评分及进度条。
- `terminal.log` 保存 Python stdout/stderr 和官方 logger；`--log-level DEBUG` 额外
  显示官方模型输入/输出、reward、done。底层 C++/Unity 直接写系统文件描述符的输出
  不一定进入 Python tee；如需这些初始化输出，启动时再用 shell `2>&1 | tee ...`。
- Habitat 每类目录保留 `images/episode_*/`、`episode_*_step_*.json` 完整原版环境日志、
  `results/episode_*_final_res.json`、`results/summary.json` 和 `config.txt`。
- NAV 保留原版图片、逐步 `episode_*_step_*.json`、`results/episode_*_final_res.json`、
  `results/summary_all.json` 和 `config.txt`，具体文件命名由官方环境决定。
- 根目录 `config.json` 是额外的运行元数据：官方配置、源码哈希、轨道、起始时间、
  Git 状态和运行时修改；`summary.json` 是额外的类别汇总，原版结果文件不被改写。
- `audit/api_requests.jsonl` 记录官方 SDK 收到的参数和完整提示；图像 data URL
  无损拆为 `audit/input_images/` 字节文件、header 和哈希，可以重建。
  Gemini 的 Pydantic schema 类记录其类名及 JSON schema；这是 SDK 参数审计，不是抓取 HTTP wire。
- `audit/api_responses.jsonl` 记录 API 返回对象（包括 content、parsed、model、usage）；
  `api_errors.jsonl` 记录异常类型、HTTP 状态和脱敏后的 provider message/code/type/param；
  `episodes.jsonl` 关联实际任务和调用起点。
  API transport 内部重试不单独计数。所有审计文件都不会回读到模型。

完整原版环境 info 仅用于评估/保存，模型只看到该轨道允许的输入。
不要把官方结果里的真实 `last_action_success` 误认为已经泄露给模型。

## 旧结果与验证记录

此前 `closed_feedback_v1` 的独立循环已移除。旧运行
`running/closed_feedback/gpt54mini_hab_rgb_pilot/` 保持原样：三帧、强制逐动作、不同解析/
重试/任务顺序，两个任务成功率 50%。只能作为旧协议 pilot，不能与 v2 混合汇总。
也不能直接将旧 w/ 主表数字与 v2 的 w/o 数字做严格配对，应核对模型版本及任务范围后重跑。

单元测试直接加载本项目官方 evaluator/planner 的真实函数，验证官方多动作计划、
失败评分和原版日志调用仍保留；仅严格 RGB 轨道不按隐藏失败标志重规划。
同时验证官方 YAML 默认值、提示历史隔离、除 schema 兼容约束外 API 参数不变、源码变化时停止和目录冲突处理。

2026-09-08 无 API 仿真验证产物：
`running/test/stock_v2_hab_smoke/` 与 `running/test/stock_v2_nav_smoke/`，各两个 Base 任务、
seed 0、每任务两步、固定 action 0 两动作计划，API 调用 0。
它们已生成官方原始环境日志、结果文件和逐步终端输出；不代表模型性能。

最终回归：`python -m pytest -o addopts='' -q`，**304 passed**。
两环境另各跑了 `rgb_only` / `full` 单任务四步配对检查：seed 0、默认单帧、
固定两动作计划，每次均有两次 planner 调用、4 次执行动作、API 调用 0，成功率 0。
结果位于 `running/test/stock_v2_{hab,nav}_{rgb_only,full}_validation/`。
实际后续请求验证：RGB-only 历史无反馈插值，full 有反馈；两轨均保存了
原版环境日志、结果文件和终端逐步输出。未运行付费模型，API 的 schema 兼容性仍需
用户启动真实小样本验证；遇到中转平台不支持官方请求格式会报错，不自动降级。

2026-09-08 全量运行中断排查：`gpt-5.4-mini_eb-hab_v2_full` 已完成 Base 50 个、
Common Sense 15 个任务，第 16 个 Common Sense 任务中的调用 247 返回 HTTP 400。
旧日志只有 BadRequestError/400，未保存 provider 原因，不能判断为密钥或 schema 问题。
随后 colorama atexit 写入已关闭的 Tee 日志产生二次异常；现已修复 Tee 的关闭后写入/
flush，并增加脱敏 API 错误详情、failure.json 状态码及终端错误落盘。
本次修复的 11 项相关测试通过；原始中断的 HTTP 400 根因仍待明确错误详情确认。

续跑验证（2026-09-08）：真实无 API fixtures 位于 `running/test/resume_v2_hab_fixture/`
和 `running/test/resume_v2_nav_fixture/`。两者先用固定策略完成两个任务，各两步，再在隔离
fixture 中移除第二条结果模拟中断。控制器只重跑第二个任务，成功合并为 2/2；
第一次结果的源文件与哈希保留，任务身份检查通过。该测试不消耗模型 API，也不代表性能。
实际全量目录 dry-run 检测为 65/300，从 Common Sense 第 16 个任务开始。

### 2026-09-08 后续中断：明确的 schema 错误

`_resume/attempt_202609081828/audit/api_errors.jsonl` 连续四次返回 HTTP 400，
code 为 `invalid_json_schema`，param 为 `text.format.schema`，明确要求
`additionalProperties: false`。这次错误不是连接超时；关闭代理转发本身不能修复该请求。
[OpenAI Structured Outputs 文档](https://developers.openai.com/api/docs/guides/structured-outputs)
说明了结构化输出对象的这一约束。

适配层现在只对发送给 SDK 的字典 schema 副本补齐根对象和动作对象的
`additionalProperties: false`，保留名称、字段、required、描述和其他请求参数；
不修改官方源码/全局 schema，不添加额外提示，不降级为自由文本。
Gemini 的 Pydantic schema 仍交给 SDK 转换。该兼容处理对 full/rgb_only 两轨一致。
明确的无效 schema 错误不再触发请求重试或进程重启；连接/超时和其他暂时性错误
仍按原有限重试策略处理。`failure.json` 也保存脱敏后的 provider 原因。

新运行 `config.json` 和请求审计记录 `schema_compatibility=closed_objects_v1`。
续跑索引逐任务保存此版本，旧运行记为 `stock`；汇总增加
`schema_compatibility_counts` 和 `mixed_schema_compatibility`。旧结果原样保留，
但修复前后混合汇总应注明 schema 约束发生过变化，不能描述为请求配置完全一致的
整批实验；严格配对比较应使用相同兼容版本。

关闭服务器转发后，如当前 shell 还保留代理环境变量，可先清理再续跑（保留已导出的 key）：

```bash
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
bash /root/max/VISTA-Skill/scripts/resume_closed_source_wo_feedback.sh
```

验证：`python -m pytest -o addopts='' -q tests/test_closed_feedback.py tests/test_feedback_resume.py`
为 **23 passed**，覆盖官方嵌套 schema、副本不污染、错误分流、断点和混合版本来源。
实际目录只读 dry-run 确认为 **124/300**，下一任务为 Complex Instruction 第 25 个。
本次没有调用付费 API，网关端实际接受情况仍需真实启动确认。

### 2026-09-08 按用户要求整合结果并改为原目录续写

已先将 `gpt-5.4-mini_eb-hab_v2_full/_resume/` 中首个完整落盘的 59 个任务
（含成功和正常结束的失败任务）复制回主目录。共 1,101 个结果/轨迹/图片文件，
复制前后 SHA-256 一致；主目录现在有 Base 50、Common Sense 50、Complex Instruction 24，
合计 124/300 个完整任务。旧 `_resume` 的结果和 API 审计保留作为来源，不能重复计数。
整合清单、元数据备份及路径/哈希位于
`running/migrations/closed_feedback_consolidation_202609081850/manifest.json`。
主目录发生冲突的中断残留已归档到 `_interrupted/closed_feedback_consolidation_202609081850/`。
这次只调整存储位置，没有重跑真实模型或改动任务分数。

`scripts/consolidate_closed_feedback.py --run-dir <原目录>` 可单独执行旧格式整合，
支持 `--dry-run`，重复执行不会重复导入。续跑控制器发现旧 `_resume` 中仍有尚未整合的
已接受结果时，也会先执行整合。

脚本新逻辑见上文“中断后续跑”。验证命令：

```bash
python -m pytest -o addopts='' -q tests/test_closed_feedback.py tests/test_feedback_resume.py tests/test_feedback_consolidation.py
```

**27 passed**。另以真实 Habitat 和 Navigation 固定策略验证，产物分别位于
`running/test/inplace_resume_hab_202609081855/` 与 `running/test/inplace_resume_nav_202609081855/`。
各自保留任务 1、模拟任务 2 的损坏结果；最终续跑 session 只执行任务 2、seed 0、两步、
一次固定策略调用、API 费用 0，两组均恢复到 2/2，任务成功率均为 0，不代表模型性能。
任务 1 的哈希保持不变，任务 2 直接写回主目录，损坏结果已归档；再次启动完整目录
没有产生新 session。首轮测试发现运行时函数的全局依赖绑定遗漏，已停止测试并修复，
上述结果来自修复后的最终 session；首轮测试日志保留供诊断。

### 2026-09-09 Gemini schema 兼容与分会话运行

Gemini 3 Flash Preview 在 0/300 时、Gemini 3.5 Flash 在 3/300 时均出现
`generation_config.response_schema` 不接受 `$defs`/`$ref` 的 HTTP 400。
适配层现在对这两个模型使用原 Pydantic 模型的子类，仅覆盖 JSON schema 生成，
将本地引用展开；保留字段、约束和 SDK 的 `beta.chat.completions.parse` 校验路径，
不修改官方 schema 类或 planner。递归/外部引用直接拒绝，不丢弃约束。
新 session 的版本记录为 `gemini_inline_refs_v1`，旧结果及其 schema 来源版本保留。
明确的 Google response_schema/Unknown name 错误现在归类为不可通过重试修复。

针对两个模型用真实安装的 OpenAI SDK 加本地 HTTP mock 验证：线上请求格式没有
`$defs`/`$ref`，仍能解析为原 ActionPlan 类型，其余请求参数保留。
相关测试 **30 passed**。Gemini 3 Flash 已在 `gemini3-hab` 会话实际取得成功 API 响应并
产出任务结果。`closed-feedback` 专用于 Gemini 3.5 的 Habitat→Navigation 序列。
用户明确授权两个监看子智能体，在进程退出后对暂时性故障进行有限断点重启；
仍保留首个完整结果，不因任务失败而重跑。

### 2026-09-10 Qwen3-VL-8B-Instruct 同设置对照

新增 `--provider qwen`，必须明确提供推理服务器 `--base-url`。保留官方
RemoteModel 的 `_call_qwen7b` 路径及消息转换、temperature=0、max_tokens=4096，
使用同一 `stock_feedback_v2/rgb_only` 反馈适配；不加入 Skill、teacher 或演化。
默认单帧、500 分辨率、seed 0，Habitat 10-shot、NAV 3-shot，动作预算由官方环境决定。
服务无鉴权时使用 SDK 所需的占位 key；有鉴权时导出 `QWEN_API_KEY`，不复用 OpenLux key。

已确认独立 endpoint（均返回 `Qwen/Qwen3-VL-8B-Instruct`、max_model_len=16384）：

| 环境 | endpoint | tmux |
|---|---|---|
| EB-Habitat | `http://192.168.1.185:8000/v1` | `qwen-hab` |
| EB-NAV | `http://192.168.1.173:8001/v1` | `qwen-nav` |

启动或恢复命令（默认各先验证 Base 2 个任务，正常结束即可进入全量，不按成功率筛选）：

```bash
bash /root/max/VISTA-Skill/scripts/run_qwen_feedback_experiment.sh eb-hab
bash /root/max/VISTA-Skill/scripts/run_qwen_feedback_experiment.sh eb-nav
```

模型服务器分开，仿真器仍在本实验机运行。pilot 位于
`running/test/Qwen3-VL-8B-Instruct_{eb-hab,eb-nav}_feedback_pilot/`，全量位于
`running/closed_feedback/Qwen3-VL-8B-Instruct_{eb-hab,eb-nav}_v2_full/`（沿用反馈对照 cohort 的位置）。
两组均独立注册任务、配置和 endpoint identity，直接使用同一续跑控制器补齐原目录结果。
全量每组300个任务，不把pilot结果计入全量统计。单独入口
`scripts/run_qwen_wo_feedback.py` 支持 `--env`、`--base-url`、`--model`、`--output`、
`--episodes` 和 `--dry-run`；在启动仿真前检查模型列表。

网络连通性由 DHCP 临时地址和两条目标路由恢复：实验机原有IPv4网关不可达，
并非只是沙箱限制。保留原默认路由、IPv6与原地址，DHCP客户端持续续租。
恢复记录和自定义脚本备份在 `running/test/qwen_feedback_network_20260910/`。
不要在实验期间删除临时地址或停止该DHCP客户端；两组运行结束且无其他任务依赖时，
可用该记录中的 PID/租约信息释放临时租约并清理仅本任务新增的地址、目标路由和自定义脚本。
未修改 AppArmor 策略；使用系统允许的 dhclient 自定义脚本位置。

验证：相关测试 **33 passed**；两个真实模型pilot均已取得API响应并执行官方动作。
接口公布的信息不能独立证明两服务的权重量化/GPU/TP完全一致，服务器部署沿用用户既有配置。
