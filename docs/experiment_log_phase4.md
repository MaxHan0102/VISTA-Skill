# VISTA-Skill Phase 4 实验日志

本文件记录 2026-08-27 开始的人工目标 Skill（post-hoc oracle target）验证。它不修改
Phase 1--3 的冻结结论，也不把人工读取测试轨迹得到的 Skill 冒充为自动演化结果。

## P4-O1：从历史 No Skill / Static Skill 轨迹合成目标 Skill

### 研究问题

跳过 8B executor 自主优化过程，由研究者综合历史轨迹直接写出一个符合 VISTA 五字段结构的
“目标 Skill”。检验这个目标是否能形成明显强于 No Skill 和原始 Static Shared Skill 的执行上界，
并据此定义后续自动演化方法应该学到的行为。

### 信息边界与结论边界

- Skill 编写读取了 EB-HAB 与 EB-NAV 的 `official_test/base` 历史轨迹，因此是
  **post-hoc oracle diagnostic**，存在 source-split contamination。
- 不允许将它报告为 controlled evolution、held-out final-test 或自动 Skill learning 结果。
- 首轮验证选择未参与 Skill 合成的 `official_test/common_sense` 子集；这避免同 episode
  直接回放，但不能消除研究者看过相邻 benchmark 分布带来的间接先验。
- frozen evaluation 期间 teacher、attribution、patching 均关闭；只比较 prompt 中是否注入
  No Skill、原 S0 或人工 Target Skill。

### 冻结 source trajectories

| 环境/臂 | artifact | SHA256 | episodes | success | progress | invalid action ratio |
|---|---|---|---:|---:|---:|---:|
| EB-HAB No Skill | `running/pilot/eval/no_skill.jsonl` | `4a61c0cf…` | 20 | 0.550 | 0.633 | 0.386 |
| EB-HAB Static S0 | `running/pilot/eval/static_shared_skill.jsonl` | `6ac9f987…` | 20 | 0.450 | 0.533 | 0.421 |
| EB-NAV No Skill | `running/vista_skill/nav_official/base/no_skill/events.jsonl` | `4ac7cee2…` | 60 | 0.633 | 0.000 | 0.267 |
| EB-NAV Static S0 | `running/vista_skill/nav_official/base/static_shared_skill/events.jsonl` | `96f77629…` | 60 | 0.600 | 0.000 | 0.226 |

完整可复现统计写入 `running/target_skill_oracle_v1/source_analysis.json`。这里采用当前磁盘上的
E6 对照轨迹；它与更早 E5 的 EB-HAB 小样本方向不同，这正是本实验不把 S0 均值当作可靠优化
信号、而改为读取 episode-level failure 的原因。

### 轨迹归纳出的目标行为

EB-HAB 的主要可修复模式：

1. Navigation 成功只证明到达 receptacle，不能证明目标物体在那里；Pick 前必须有目标可见或
   near 证据。
2. `not near` 后禁止在相同位置重复 Pick；应标记已搜索位置并切换到新的 receptacle。
3. 保持指令中的精确类别、数量和目的地，禁止 mug→cup、orange→can 等替代。
4. Pick/holding 成功后立即切换 delivery，禁止继续 Pick；Place 必须对准原指令目的地。
5. remove/detach 不是“拿在手上即完成”，而是从源位置移走并放到不同的有效 receptacle。

EB-NAV 的主要可修复模式：

1. 历史两臂几乎所有 episode 都生成至少一次 3+ action 的开环计划；均值最大开环长度分别为
   4.77 / 5.02 actions。
2. 每次 planner call 强制只输出一个 primitive action，再依据新 distance 闭环决策。
3. action success 不等于 progress；只有数值 distance 下降才延续该方向。
4. distance 上升立即反向/回退；distance 不变或 action invalid 时禁止重复相同平移。
5. 数值 distance 优先于“看起来很近”，在 1.0m 阈值附近只做单步探测，避免越过最佳点。

### 目标 Skill artifacts

| 环境 | Skill ID | rendered words | Skill SHA256 | artifact SHA256 |
|---|---|---:|---|---|
| EB-HAB | `target_habitat_rearrangement_oracle_v1` | 511 | `5e32f5ec…` | `59e53743…` |
| EB-NAV | `target_feedback_navigation_oracle_v1` | 367 | `3f026dc2…` | `a4ab4e7c…` |

Canonical artifacts 与完整明文位于 `running/target_skill_oracle_v1/artifacts/manifest.json`。代码入口是
`vista_skill.skills::{target_habitat_skill_v1,target_navigation_skill_v1}`，构建脚本是
`scripts/build_target_skill_artifacts.py`。

### 预注册验证设置（运行前冻结）

- endpoint：`http://192.168.1.185:8000/v1`，Qwen/Qwen3-VL-8B-Instruct，2026-08-27
  serving-contract probe 6/6。
- arms：`no_skill`、`static_shared_skill`、`target_skill`。
- environments：EB-HAB、EB-NAV。
- subset：各自 `official_test/common_sense` 的 stock order 前 20 episodes。
- rollout seed：0；temperature：0；resolution：500；TP=1。
- EB-HAB n-shot=10，EB-NAV n-shot=3；同环境三臂完全一致。
- primary：Task Success；secondary：Task Progress、composite score、invalid action ratio、steps。
- paired reporting：target-minus-control bootstrap 95% CI、success wins/losses、exact McNemar p。
- 这是单 seed、20-episode/arm 的 diagnostic；置信区间与不确定性必须同时报告。

执行命令由 `scripts/run_target_skill_validation.py` 固化，每臂保存 command、console log、events、
summary 和 runtime manifest。结果在实验完成后追加，禁止根据中途结果改写 v1 Skill。

### 执行记录与故障账本

- 实验名：`P4-O1_target_skill_oracle_v1_common_sense_seed0`。
- EB-HAB 三臂位于
  `running/target_skill_oracle_v1/validation_common_sense_seed0/eb_hab/common_sense/`。
- 首次切换 EB-NAV 时，harness 错误沿用了 `max_embench` Python，因缺少
  `ai2thor.platform` 在 0 个 episode 前退出（exit=1）。失败 console 与 runtime manifest 原样保留在
  `validation_common_sense_seed0/eb_nav/common_sense/no_skill/`，没有把它计入结果。
- 修复 harness，使 `--nav-python` 默认为 `max_embench_nav` 后，从全新目录
  `validation_common_sense_seed0_nav_resume1/` 重跑 EB-NAV 三臂。有效三臂数据集 SHA256 均为
  `3e7d2cb4…`，seed、模型、顺序和配置匹配。
- 统一分析：`running/target_skill_oracle_v1/validation_common_sense_seed0_analysis.json`
  （file SHA256 `338f296e…`）；NAV resume manifest 的 file SHA256 为 `cb85714e…`。

### 冻结验证结果

以下均是每臂 20 episodes；invalid 是按全臂 `sum(invalid)/sum(env_steps)` 聚合。

| 环境 | arm | success | progress | composite | invalid | mean env steps | mean planner calls | runtime (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| EB-HAB | No Skill | 0.150 | 0.1875 | 0.1193 | 0.4450 | 19.55 | 9.55 | 901 |
| EB-HAB | Static S0 | 0.250 | 0.2750 | 0.2258 | 0.3976 | 16.35 | 7.90 | 779 |
| EB-HAB | **Target v1** | **0.250** | **0.3250** | **0.2396** | **0.3516** | 19.20 | 8.65 | 915 |
| EB-NAV | No Skill | 0.550 | 0.0000 | 0.3637 | 0.2508 | 15.15 | 6.80 | 670 |
| EB-NAV | Static S0 | **0.650** | 0.0000 | **0.4365** | 0.2411 | 14.10 | 6.45 | 671 |
| EB-NAV | **Target v1** | 0.600 | 0.0000 | 0.4058 | **0.1825** | 14.25 | 14.25 | 1124 |

Macro-average success 为 No Skill 0.350、Static S0 0.450、Target v1 0.425。也就是说，Target
相对 No Skill 为 **+7.5 percentage points**，但相对 Static S0 为 **-2.5 points**；它没有达到
预期的“大幅提升”上界。

配对不确定性同样不支持 primary-metric 提升：

- EB-HAB Target − No Skill：success `+0.10`, bootstrap 95% CI `[-0.15, 0.35]`，
  wins/losses/ties=`4/2/14`，exact McNemar `p=0.6875`；progress `+0.1375`
  `[-0.10, 0.375]`；composite `+0.1203` `[-0.1307, 0.3712]`。
- EB-HAB Target − Static：success `0.00` `[-0.25, 0.25]`，`3/3/14`，`p=1.0`；
  progress `+0.05` `[-0.175, 0.275]`；composite `+0.0138`
  `[-0.2346, 0.2639]`。
- EB-NAV Target − No Skill：success `+0.05` `[-0.20, 0.30]`，`4/3/13`，`p=1.0`；
  composite `+0.0421` `[-0.1386, 0.2242]`。
- EB-NAV Target − Static：success `-0.05` `[-0.35, 0.25]`，`4/5/11`，`p=1.0`；
  composite `-0.0307` `[-0.2460, 0.1864]`。

### 机制读数

Target v1 并非完全无效；它改善了被明确编码且 8B 容易遵循的局部行为：

1. EB-HAB 全臂 invalid ratio 从 No Skill 的 0.4450 降至 0.3516，progress 从 0.1875 升至
   0.325。它相对 No Skill 新救回 episode 29/35/40/42，但回退 episode 2/4；相对 Static
   则是 3 wins / 3 losses。
2. EB-NAV 的单步约束在原始轨迹中达到 20/20 episodes：Target 的 mean maximum open-loop
   plan length 恰为 1.00，而 No Skill/Static 为 4.75/4.85，且后两者均 20/20 出现 3+ action
   plan。Target 的聚合 invalid ratio 也降至 0.1825。
3. 这些局部改善没有转化为更高 success。Target 的 NAV mean minimum distance 为 1.287m，
   反而差于 No Skill 1.167m 和 Static 1.075m；单步反馈能防止 stale plan，却不能替代初始
   搜索方向和视觉语义判断。Target 需要每个环境动作都重新调用 planner，平均 planner calls
   从 6.80/6.45 增至 14.25；runtime 比两个对照约高 68%。
4. EB-HAB 中，长文本规则存在明显 instruction-following gap。例如 Target 在 episode 2 仍从
   empty-handed Place 开始并重复失败 Pick；episode 4 把“cut paper”绑定成 knife 而非 benchmark
   所需 scissors；episode 29/35/40/42 又表明精确 search-delivery checklist 在部分 episode
   确实能够救回任务。规则本身合理不等于 8B 会稳定执行。

### 结论与对演化方法的约束

P4-O1 的结论是 **未验证“大幅能力提升”，但验证了两个可控机制信号**：结构化 Target Skill
能降低无效动作，并能把 NAV 开环动作串完全改造成单步闭环；然而 monolithic prose Skill
不能稳定解决目标语义 grounding、搜索方向选择和规则遵循，因此不是期望中的强 oracle。

后续 Skill 演化应以此作为反例和设计参考：

1. 不以“规则写得更完整”作为代理目标；candidate gate 必须看 paired Task Success、subgroup
   regression、invalid burden 与 executor/planner cost。
2. 优化 action-local、短且有优先级的决策规则；对 `holding -> 禁 Pick`、`not-near -> 禁同地
   重试` 等可判定约束，优先编译成 non-invasive action gate，而不是仅依赖 prompt 遵循。
3. 将 common-sense referent grounding 与 transition policy 分开归因；先把 “sports object”、
   “object for cutting paper” 等绑定到 benchmark object category，再锁定身份，避免把“精确绑定”
   误实现成字面词匹配。
4. NAV 使用 adaptive horizon：在 distance 上升、停滞、blocked 或接近 1m 时强制单步；在方向
   已连续验证改善且距离较远时允许短 burst。这样保留反馈纠错能力，又避免 Target v1 的 2.2x
   planner-call 开销。
5. evolution update 应是 field-local patch，并保留可执行 adherence check；一次性写满 511 words
   的 HAB Skill 容易稀释关键约束。Teacher cost 本实验为 0，automatic evolution calls 也为 0，
   不能把人工目标的成本记作方法成本或自动演化收益。

因此，Target v1 保持冻结并作为 `post-hoc target/reference` 留存；不在 common_sense 结果之后
制作 v1.1 来追分。若继续实验，应预注册一个从 base/dev 演化的 concise + compiled-gate v2，
再使用未查看的 episode/subset 做独立验证。
