---
tags:
  - AIGC
---

# VISTA-Skill 最新方案：证据解耦的视觉转移信用分配与可靠 Skill 进化

> 更新时间：2026-08-06
>
> 定位：CVPR 2027 冲刺方案与实验执行笔记
>
> 状态：研究问题和最低成本实现路径已经收敛，核心方法与主实验尚待实现和验证。

本文档是在以下材料基础上的最新汇总：

- `20260722-VISTA-GraphSkill-CoEvolution-CVPR2027方案备份.md`；
- `20260730-VISTA-Skill最终收敛方案-视觉状态转移信用分配与可靠Skill进化.md`；
- `20260730-VISTA-Skill问题定义与Skill-Pro对比核心Insights.md`；
- Skill-Pro 在 EmbodiedBench-Habitat 上的官方协议对齐复现；
- EmbodiSkill、SkillOpt、Skill-Pro、HDSO、SkillAudit、SSO、HiMPO 等相关工作与开源实现；
- 当前论文主表、硬件条件和可接受的租卡上限。

这里的“最新方案”不代表实验结论已经成立。文中会明确区分：

- 当前论文必须验证的核心；
- 为降低成本而采用的实现收缩；
- 只有核心假设成立后才值得做的扩展；
- 实验失败时的降级与 benchmark 备选路线。

---

## 0. Executive Summary

### 0.1 一句话研究问题

> **在部分可观测的具身执行中，哪些动作级视觉转移应被固化为可复用程序化 Skill，哪些只应触发当前信念修复或拒绝长期写入？**

英文推荐表述：

> **Which action-level visual transitions should be consolidated into reusable procedural skills under partial observability, and which should instead trigger belief repair or no persistent update?**

推荐问题名称：

> **Visual Transition Credit Assignment for Reliable Skill Evolution**
>
> 面向可靠 Skill 进化的视觉转移信用分配

### 0.2 一句话 Motivation

> **任务失败不是 Skill 错误的可靠证据：失败可能来自视觉误判、执行器没有遵循正确规则、遮挡或偶发动作失败；未经归因就反思整条轨迹，会把一次看错或偶发失败固化成长期错误 Skill。**

### 0.3 一句话 Method

> **VISTA-Skill 让当前 Skill 在结构化谓词空间中给出动作后的预期变化，同时从动作前后图像和环境反馈中独立提取证据支持的变化；系统比较二者并先判断应该刷新当前 belief、修改 Skill，还是 abstain，只有重复出现且通过配对验证的 Skill 缺陷才会形成字段级长期补丁。**

### 0.4 当前最重要的项目决策

1. **EmbodiSkill 是主 baseline。** Skill-Pro 降为 motivation、诊断和附录复现，不再作为论文性能主线。
2. **性能是主指标。** Attribution、harmful update、teacher cost 等指标用于证明性能提升来自正确机制，而不是取代性能。
3. **不先实现完整 graph–skill co-evolution。** P0 使用稀疏谓词 ledger、固定动作 schema 和三路 update routing。
4. **不使用 Skill-Pro PPO/logprob gate。** 使用低成本的 bounded patch 与 staged paired gate。
5. **核心实验由 Qwen3-VL-8B-Instruct + 4090 完成。** RTX PRO 6000 96G 只用于通过 Go/No-Go 后的最终 32B scale experiment。
6. **测试时冻结最终 Skill。** Attribution、patch 和 evolution teacher 仅在 Skill acquisition/evolution 阶段运行。

### 0.5 当前方法的最小闭环

```text
共享初始 Skill S0 + 固定动作 Schema
                ↓
       Frozen VLM 执行动作
                ↓
   Skill Expected Transition
        vs. 独立 Visual Evidence
                ↓
 Visual Transition Credit Assignment
      ┌─────────┼─────────┐
 belief refresh  skill defect  abstain
                      ↓
              target → Skill field
                      ↓
              bounded field patch
                      ↓
              staged paired gate
                      ↓
              accepted Skill S'
```

---

## 1. 当前方案是如何演化过来的

### 1.1 第一阶段：Graph–Skill Co-Evolution 系统蓝图

最初方案试图同时维护：

- episode-level visual interaction graph；
- persistent action-effect graph；
- graph-grounded procedural Skill；
- expected vs. observed graph transition；
- graph 和 Skill 的跨轨迹联合更新。

这一阶段的直觉是合理的：长程具身任务需要外部状态，Skill 也应根据交互不断改进。但它有三个问题：

1. 更像一个复杂系统组合，缺少单独可定义、可证伪的新问题；
2. 没有证明完整 graph 或 coupled update 是必要的；
3. 实现和消融规模超过当前 4090 条件下的合理 P0。

因此，“Graph–Skill Co-Evolution”从问题定义降为可能的系统性质，只有 coupled update 显著优于 skill-only/independent update 时才允许重新写入贡献。

### 1.2 第二阶段：从图共进化收窄为 mismatch attribution

随后方案转向：

> 当 Skill 预测的状态变化与实际观察不一致时，判断应该修改 belief、动作知识还是 Skill。

这一版本首次把核心放到“谁应该承担错误”上，但仍有一个关键逻辑风险：

- 在 POMDP 中，动作后图像并不等于环境真实状态；
- 如果 observed transition 的生成过程看到了 Skill prediction，它可能只是重复 Skill 的假设；
- 之后的 attribution 会形成 prediction leakage 和自我验证。

因此，“actual/observed transition”被严格改写为：

> **evidence-supported transition：当前独立图像和反馈能够支持的 belief change，而不是不可见的 ground-truth state change。**

### 1.3 第三阶段：Visual Transition Credit Assignment

7 月 30 日版本将研究问题正式收敛为四类 update target：

```text
belief / action model / skill / no persistent update
```

并引入：

- Expected Branch 与 Evidence Branch 的信息隔离；
- target→field 层级归因；
- 五字段 Skill；
- recurrence、bounded patch、paired validation；
- beneficial/harmful update 的直接评测。

这一版本科学问题已经清晰，但 P0 仍包含：完整 episode belief、四种可写 memory、persistent action model、五字段归因、多个 optimizer baseline 和 coupled update，工作量仍然偏大。

### 1.4 Skill-Pro 官方协议对齐复现带来的实证修正

Skill-Pro 在 EB-HAB 的官方协议对齐版使用：

- 官方 `VLMPlanner`；
- 视觉输入；
- `habitat_system_prompt`；
- 10-shot examples；
- JSON 多动作计划；
- Qwen3-VL-8B-Instruct fp8；
- 单卡 RTX 4090 D 24GB。

复现结果为：

| 设置 | EB-HAB 六子集平均成功率 |
|---|---:|
| No Skill | 29.3% |
| Seed Only | 34.7% |
| Full Skill-Pro w/o PPO | 29.3% |

关键事实：

- 手写 seed 带来 `+5.3` 个百分点；
- 无 gate 演化相对 seed-only 带来 `-5.3` 个百分点；
- 六个子集全部退化；
- 完整 PPO 在 8B 上出现 ratio≈1、候选几乎全部拒绝；
- `prompt_logprobs` 的全词表峰值在 24GB 上导致 vLLM 不稳定；
- Skill-Pro SMDP 每步约 2–2.5 次模型调用，明显高于官方 direct planner。

这证明：

1. “未经验证的 Skill evolution 会有害”是真实且有价值的 motivation；
2. 继续租大卡救 Skill-Pro PPO 的投入产出比低；
3. 新方法应避免依赖 executor 对微小 Skill 文本变化的 logprob 敏感性；
4. 低成本的 evidence filtering 与 held-out gate 比 Non-Parametric PPO 更适合当前项目。

详细复现见：

- `../../Experiments/Skill-Pro-EmbodiedBench/docs/experiment_progress.md`；
- `../../Experiments/Skill-Pro-EmbodiedBench/docs/experiment_results.md`；
- `../../Experiments/Skill-Pro-EmbodiedBench/docs/official_alignment_changes.md`。

### 1.5 EmbodiSkill 开源情况带来的 baseline 修正

EmbodiSkill 论文在 EB-HAB 上报告：

- Qwen3-VL-8B + GPT teacher：45.33%；
- Qwen3-VL-8B + Gemini teacher：41.00%；
- Qwen3-VL-32B + GPT teacher：50.33%；
- Qwen3-VL-32B + Gemini teacher：52.33%。

它使用 1000 个 EmbodiedBench training tasks 和 10 个 revision stages。但当前公开仓库主要提供 ALFWorld 流程，EmbodiedBench 的 1000 个训练任务和完整评测实现未公开，相关 issue 仍在询问。

因此最终论文必须区分：

- **EmbodiSkill reported**：论文原始报数，仅作参考；
- **EmbodiSkill\***：在本项目统一 VLMPlanner、公开数据、teacher 和预算下的 controlled reimplementation。

论文不能把 reported number 与本项目不同协议的数字混进同一公平对比区块。

### 1.6 最新收缩：从“四种长期 memory”到“三路核心 routing”

考虑创新边界、EB-HAB 的动作定义和实现成本，当前 P0 采用：

```text
belief refresh / skill update / abstain
```

其中：

- action schema 是固定环境知识，不在主系统中在线学习；
- action-model fault 保留在 fault-injection benchmark 中；
- persistent action-model update 降为 P2；
- scene graph 降为稀疏 typed predicate ledger；
- 论文 novelty 从“多 memory 联合进化”收紧为“action-level、vision-grounded、evidence-decoupled update attribution”。

这不是放弃原问题，而是把最能被当前资源验证、又最不容易与已有工作重叠的部分提到中心。

---

## 2. 当前方案的 Motivation

### 2.1 核心现象：一次失败不等于 Skill 错了

以任务为例：

> **Instruction:** Place all apples on the TV stand.

Agent 执行：

```text
place(apple_1, tv_stand)
```

动作可能完全成功，但当前 Skill 错误地写着：

```text
Termination: Stop after placing one target object.
```

此时至少存在以下不同原因：

| 表面现象 | 真实原因 | 正确响应 |
|---|---|---|
| 看不到第二个苹果 | 遮挡/证据不足 | 继续观察，不长期更新 |
| 把两个苹果合并成一个实例 | episode belief 错误 | 修复当前 belief |
| Skill 认为一个完成即全部完成 | termination defect | 修改 Skill termination |
| Skill 已要求检查所有实例，但 executor 没执行 | execution lapse | 不改 canonical Skill |
| Place 偶发 no-op | stochastic failure | 记录但 abstain |

轨迹级反思常采用：

```text
task failure → summarize failure → rewrite skill
```

它容易把一次视觉误判、执行器走神或偶发 no-op 固化成长期规则，造成跨任务负迁移。

### 2.2 Skill Prediction 到底是什么

Skill prediction 不是生成未来图像，也不是自由描述“接下来大概会怎样”。它是结构化预期谓词变化。

例如动作前：

```text
holding(apple_1) = true
on(apple_1, tv_stand) = false
on(apple_2, tv_stand) = false
```

根据固定 action schema 和当前 Skill，预期为：

```text
holding(apple_1): true → false
on(apple_1, tv_stand): false → true
task_complete: false → true   # 来自错误 termination
```

它同时包含：

- primitive action effect；
- Skill procedure 对下一阶段的预期；
- Skill termination 对是否完成的判断。

### 2.3 Visual Evidence 到底是什么

Evidence Branch 只读取：

- 动作前图像 `x_t`；
- 实际动作 `a_t`；
- 动作后图像 `x_{t+1}`；
- 环境反馈 `f_t`；
- 动作前的局部 belief，但不读取 expected transition。

它输出三值谓词：

```text
true / false / unknown
```

例如：

```text
holding(apple_1) = false
on(apple_1, tv_stand) = true
visible(apple_2) = true
on(apple_2, tv_stand) = false
task_complete = false
```

若 `apple_2` 被遮挡，则输出：

```text
on(apple_2, tv_stand) = unknown
```

而不是因为“没看见”就写成 false。

### 2.4 核心 Insight

> **Skill 先明确预测“这一步后应该发生什么”，独立证据分支再判断“图像和反馈实际支持发生了什么”；只有二者的失配被归因为可复用 Skill 缺陷时，经验才有资格进入 Skill optimizer。**

### 2.5 与最强相关工作的边界

近期工作已经覆盖：

- Skill-Pro：semantic gradient + candidate gate；
- EmbodiSkill：skill defect vs. execution lapse routing；
- SkillOpt：bounded patch + validation；
- HDSO：falsifiable hypothesis + paired control/treatment；
- SkillAudit：paired trajectory auditing + passage-level repair；
- SSO：unlabeled comparative skill optimization；
- HiMPO：long-horizon memory-writing credit entanglement。

因此不应声称：

- 第一个可靠 Skill update；
- 第一个 paired validation；
- 第一个 memory credit assignment；
- bounded patch 或 held-out gate 本身是主要创新。

当前可防守的创新限定为：

> **动作级、视觉落地、部分可观测条件下的 update-target attribution，以及 Skill prediction 与 visual evidence 的信息隔离。**

---

## 3. 当前方案的 Method 概览

Method 建议写成五个核心模块。

| 章节 | 目标 | 核心设计 |
|---|---|---|
| 3.1 Problem Setup and Skill Initialization | 定义视觉转移和公平初始 Skill | benchmark-aware、task-agnostic 的共享 `S0`；固定 action schema |
| 3.2 Structured Skill and Sparse Evidence State | 提供 prediction/evidence 的共同语言 | 五字段 Skill；带实例、置信度和 provenance 的稀疏谓词 ledger |
| 3.3 Evidence-Decoupled Transition Modeling | 产生可信 mismatch | Expected Branch 与 Evidence Branch 严格信息隔离 |
| 3.4 Hierarchical Visual Transition Credit Assignment | 判断是否及如何修改 Skill | `belief refresh / skill update / abstain`，再做 target→field |
| 3.5 Evidence-Gated Skill Evolution | 防止长期 Skill 污染 | recurrence、bounded field patch、staged paired gate |

完整流程：

```text
S0 + Action Schema
       ↓
Frozen Planner → Action → Pre/Post Images + Feedback
       │                         │
       ▼                         ▼
Expected Transition      Evidence Transition
       └──────── mismatch ───────┘
                         ↓
             Hierarchical VTCA
        ┌────────────────┼────────────────┐
 belief refresh     skill defect       abstain
                         ↓
              activation/procedure/
            effect/termination/constraint
                         ↓
            recurrent evidence cluster
                         ↓
               bounded field patch
                         ↓
                staged paired gate
                         ↓
                    Skill S'
```

---

## 4. Method 的详细设计与具体实现方案

### 4.1 问题定义和符号

对每个动作级交互记录：

```text
e_t = (instruction, b_t, skill_t, x_t, a_t, x_{t+1}, f_t)
```

其中：

- `instruction`：当前任务指令；
- `b_t`：动作前稀疏 belief ledger；
- `skill_t`：当前 active Skill；
- `x_t`：动作前 RGB；
- `a_t`：实际执行动作及参数；
- `x_{t+1}`：动作后 RGB；
- `f_t`：环境反馈、动作是否成功、episode done/reward。

Expected Branch 输出：

```text
delta_expected_t
```

Evidence Branch 输出：

```text
delta_evidence_t, evidence_packet_t
```

VTCA 输出：

```text
target ∈ {belief_refresh, skill_update, abstain}
```

若 `target=skill_update`，继续输出：

```text
field ∈ {activation, procedure, effect, termination, constraint}
```

### 4.2 Skill 初始化标准

#### 4.2.1 原则

主设置采用：

> **benchmark-aware, task-agnostic initialization**

即：

- 可以使用公开 action interface、PDDL、动作有效性规则；
- 不能使用 final test trajectory；
- 不能为 base/common-sense/long-horizon 等测试子集手工写规则；
- 不能用最终失败诊断倒推 seed；
- 所有 controlled baselines 使用语义一致的初始 Skill；
- 固定初始和最终 Skill token budget。

#### 4.2.2 两层初始化

第一层是固定环境 schema，不作为可进化 Skill：

```text
Navigation / Pick / Place / Open / Close
preconditions / primitive effects / parameter types
```

EB-HAB adapter 从 `language_skill_set` 和公开 PDDL/动作说明构造；EB-Navigation 只需替换动作 adapter。

第二层是通用程序化 Skill `S0`：

```text
Activation:
Use for multi-step embodied tasks requiring state verification.

Procedure:
Decompose the instruction into goal predicates; maintain pending/completed
targets; check action preconditions; update the checklist only from new evidence;
re-observe when evidence is insufficient.

Effect:
Each step should make progress toward one or more unsatisfied goal predicates.

Termination:
Stop only when every required goal predicate is evidence-verified.

Constraint:
Unknown is not false; a single subgoal success is not full task completion.
```

预算建议：

- `S0`：250–350 text tokens；
- active Skill body：不超过 512 tokens；
- appendix/temporary emphasis：若保留，不超过 256 tokens；
- 每次只注入当前相关 Skill，不注入完整历史或整个 Skill pool。

#### 4.2.3 初始化敏感性实验

至少比较：

1. Minimal/empty structured Skill；
2. Spec-initialized Skill，作为主设置；
3. 可选：从固定 acquisition trajectories 自动 bootstrap 的 Skill。

若做 trajectory bootstrap，所有方法必须使用相同任务、teacher、token budget 和生成次数。

### 4.3 五字段 Skill 表示

每个 Skill 使用：

```yaml
skill_id: multi_instance_delivery
activation:
  task_pattern: all/both/multiple target instances
procedure:
  - enumerate target instances
  - maintain pending/completed checklist
  - manipulate one instance at a time
effect:
  expected_goal_progress:
    - on(target_i, destination): false -> true
termination:
  - all bound target instances are evidence-verified at destination
constraint:
  - unknown instance state cannot be treated as completed
```

五字段的作用不是增加 prompt 复杂度，而是让 attribution 和 patch 有明确边界：

- activation：何时不该用/漏用 Skill；
- procedure：步骤、顺序、checklist 或 recovery 缺陷；
- effect：Skill 对动作后结果的错误预期；
- termination：提前结束或无法结束；
- constraint：安全、前置条件或禁止操作。

实现上保留两个视图：

- canonical JSON/YAML：供程序解析、diff 和审计；
- compact natural-language view：供 frozen VLM 执行。

### 4.4 稀疏 typed predicate ledger

不实现全场景 dense graph。每个 episode 只维护与当前指令、active Skill 和最近动作有关的局部谓词。

推荐记录：

```yaml
- predicate: holding(apple_1)
  value: true
  confidence: 0.96
  source: env_feedback
  evidence_id: ep12_step4_feedback
  timestamp: 4

- predicate: on(apple_2, tv_stand)
  value: unknown
  confidence: 0.35
  source: visual_pair
  evidence_id: ep12_step4_prepost
  timestamp: 4
```

每个 predicate 至少包含：

- instance-level arguments；
- `true/false/unknown`；
- confidence；
- provenance/source；
- evidence ID；
- timestamp/validity；
- task relevance。

原则：

- 暂时不可见不能直接改为 false；
- 多实例对象保留数字 ID；
- 不允许清洗过程删除 `apple_1/apple_2` 中的数字；
- 旧证据可以过期，但不能无来源覆盖新证据；
- ledger 是 shared interface，不是 novelty 本身。

### 4.5 Rollout 与数据记录层

复用当前 Skill-Pro-EmbodiedBench 中已经对齐的官方 `VLMPlanner`：

- `habitat_system_prompt`；
- 10-shot textual examples；
- JSON multi-action plan；
- `language_only=False`；
- visual observation；
- environment feedback。

每个 primitive action 保存：

```python
TransitionRecord(
    episode_id,
    task_id,
    instruction,
    plan_id,
    step_id,
    pre_image_path,
    post_image_path,
    action_id,
    action_text,
    active_skill_id,
    env_feedback,
    last_action_success,
    reward,
    done,
    pre_ledger_snapshot,
)
```

需要记录：

- 数据集 commit/hash；
- environment seed；
- executor/teacher 精确模型版本；
- prompt template hash；
- Skill version；
- token usage 和 wall-clock；
- raw response，便于事后复查。

### 4.6 Expected Transition Branch

#### 4.6.1 目标

输出当前 Skill 和 action schema 所预期的结构化变化，不生成未来图像。

#### 4.6.2 输入

```text
pre_ledger + active_skill + executed_action + fixed_action_schema
```

#### 4.6.3 输出

```yaml
predicted_delta:
  - predicate: holding(apple_1)
    before: true
    after: false
    source: action_schema
  - predicate: on(apple_1, tv_stand)
    before: false
    after: true
    source: skill.effect
  - predicate: task_complete
    before: false
    after: true
    source: skill.termination
```

#### 4.6.4 最低成本实现

优先使用确定性 compiler：

1. 将 action ID 映射到 action type 和参数；
2. 从固定 schema 填充 primitive preconditions/effects；
3. 从 active Skill 的 effect/termination 字段补充 procedural prediction；
4. 输出统一 predicate delta。

P0 不为 expected branch 增加一次 VLM 调用。若 Skill 中的自然语言无法编译，在 Skill 接受时由 teacher 一次性生成结构化 JSON，而不是每步重新解析。

### 4.7 Evidence-Supported Transition Branch

#### 4.7.1 信息隔离

Evidence Branch 允许看到：

```text
pre_ledger, executed_action, pre/post images, env feedback
```

禁止看到：

- expected transition；
- Skill effect/termination prediction；
- attribution target；
- candidate patch；
- parent/candidate validation result。

这是防止自我验证的核心机制。

#### 4.7.2 Rule-first 提取

第一层使用零额外模型成本的规则：

- `last_action_success`；
- environment feedback parser；
- action ID/type；
- done/reward；
- 明确的 holding/proximity/open/close feedback；
- 相邻 step 的 action history。

例如：

```text
feedback = successfully picked apple_1
→ holding(apple_1)=true

feedback = cannot place because not near tv_stand
→ last_action_success=false
→ near(tv_stand)=false or unknown
→ on(apple_1, tv_stand) unchanged
```

#### 4.7.3 Visual fallback

只有以下情况才调用本地 Qwen3-VL-8B：

- feedback 不足以判断 goal-relevant predicate；
- 需要比较动作前后实例状态；
- 需要判断 visible/on/in/open 等视觉关系；
- 可能存在多实例混淆；
- termination 依赖视觉确认。

视觉 prompt 使用两个图像和紧凑 JSON schema，输出：

```json
{
  "observations": [
    {
      "predicate": "on(apple_1, tv_stand)",
      "value": "true",
      "confidence": 0.88,
      "evidence": "apple_1 is visible on the top surface after the action"
    },
    {
      "predicate": "on(apple_2, tv_stand)",
      "value": "unknown",
      "confidence": 0.31,
      "evidence": "apple_2 is not visible in the post-action view"
    }
  ]
}
```

#### 4.7.4 视觉分支的校准

必须准备：

- simulator-assisted labels，仅用于评测；
- 200–300 条人工审计样本；
- oracle/noisy evidence ablation；
- confidence calibration；
- evidence coverage 与 selective risk。

若 visual predicate extraction 本身不可靠，不能继续宣称 VTCA 成立。

### 4.8 Mismatch 构造

对每个 goal-relevant predicate 比较：

```text
expected value vs. evidence-supported value
```

Mismatch 类型：

```text
contradiction: expected=true, evidence=false
unsupported: expected=true, evidence=unknown
unexpected_change: no expected change, evidence reports change
missing_progress: expected progress, evidence reports unchanged
termination_conflict: skill_complete=true, task evidence incomplete
```

注意：

- `unknown` 不是 contradiction；
- primitive action 成功但 task completion 错误，应定位到 Skill procedure/termination，而不是 action model；
- mismatch 只描述差异，不直接决定修改对象。

### 4.9 Hierarchical Visual Transition Credit Assignment

#### 4.9.1 第一层：update target

主系统三类：

```text
belief_refresh / skill_update / abstain
```

推荐 rule-first decision：

1. evidence coverage 不足或关键谓词为 unknown → `abstain`；
2. 新证据只推翻当前 episode 的实例状态，且 Skill 规则仍成立 → `belief_refresh`；
3. 同一 Skill 预测在多个独立上下文中被高置信证据重复否定 → `skill_update`；
4. Skill 已包含正确规则但 executor 没遵循 → `abstain: execution_lapse`；
5. 单次 no-op/stochastic failure → `abstain: stochastic`；
6. 规则无法确定时，才调用 constrained attribution teacher。

`abstain` 内部保留：

```text
insufficient_evidence / execution_lapse / stochastic_noop / ambiguous
```

#### 4.9.2 第二层：Skill field

只有 `skill_update` 进入：

```text
activation / procedure / effect / termination / constraint
```

输出必须引用：

- implicated Skill field；
- mismatch predicate；
- evidence IDs；
- independent support count；
- proposed edit scope；
- confidence。

示例：

```json
{
  "target": "skill_update",
  "field": "termination",
  "reason": "the skill predicts task completion after one object is placed",
  "evidence_ids": ["ep12_s4", "ep19_s6"],
  "support_count": 2,
  "confidence": 0.91
}
```

### 4.10 Evidence aggregation 与 recurrence

不能因一次事件直接改长期 Skill。将事件按以下键聚合：

```text
skill_id + field + mismatch_type + task_pattern + object/receptacle context
```

独立支持要求：

- 至少来自两个独立 task/episode；
- 不能只是同一事实被多次转述；
- 尽量跨 object instance 或 scene；
- evidence ID 不重复；
- high-confidence contradiction 优先；
- 低置信或 unknown 只进入 audit ledger。

初始阈值建议：

```text
support_count >= 2
mean_evidence_confidence >= 0.75
attribution_confidence >= 0.70
```

这些阈值必须在 acquisition/dev 上校准，不能根据 final test 调整。

### 4.11 Bounded Field-Level Patch

输入：

```text
current field + evidence cluster + allowed edit operations + token budget
```

允许：

- add one condition；
- replace one statement；
- delete one contradicted statement；
- append one short recovery/verification step。

禁止：

- 重写整个 Skill；
- 修改未被归因的字段；
- 引入 evidence 中没有支持的新事实；
- 增加测试任务专属 object name；
- 超过 token budget；
- 同时大幅修改 procedure 和 termination。

候选 patch 必须输出 diff 和 provenance：

```diff
- Stop after placing one target object.
+ Stop only after every bound target instance is evidence-verified at the destination.
```

### 4.12 Staged Candidate Gate

#### Stage 0：静态检查

- schema/JSON 可解析；
- field scope 合法；
- token budget；
- 无重复或矛盾规则；
- 所有修改绑定 evidence IDs。

成本：0 environment episode。

#### Stage 1：transition consistency

在 cached transition events 上检查：

- candidate 是否修复目标 mismatch；
- 是否对已正确事件产生新矛盾；
- termination/constraint 是否可执行；
- 是否过度泛化。

成本：主要是文本/规则检查，可选一次小 teacher call。

#### Stage 2：small paired proxy

在约 10 个固定 dev tasks 上进行：

```text
parent Skill vs. candidate Skill
```

固定 executor、temperature、task order 和 seed，记录：

- paired success；
- task progress；
- invalid actions；
- premature termination；
- token/call cost。

#### Stage 3：finalist paired validation

只有少量 finalist 进入约 30 个 paired dev episodes，并检查：

- overall improvement；
- protected subgroup regression；
- paired bootstrap lower confidence bound；
- worst-case category degradation。

#### Stage 4：frozen audit

Audit split 不参与候选选择，只用于事后定义：

```text
beneficial update / neutral update / harmful update
```

### 4.13 Accepted Skill、版本和回滚

每次接受保存：

```yaml
parent_version:
candidate_version:
changed_field:
patch_diff:
evidence_ids:
selection_metrics:
audit_metrics:
teacher_model:
prompt_hash:
task_split_hash:
```

保留完整 lineage，但执行时只注入当前版本，不注入 lineage 文本。

如果后续 audit 发现显著退化，可以回滚到 parent。回滚是工程保护，不作为主要创新。

### 4.14 测试时执行

最终测试阶段：

- 冻结 Skill `S*`；
- 冻结 Qwen3-VL executor；
- 禁止 attribution、patch 和 teacher；
- 不在 final test 上选择 epoch/candidate；
- 只注入 active Skill 和可选紧凑 ledger；
- 不增加 Skill-Pro 式 per-step termination/meta-controller 调用。

这保证性能提升来自训练阶段获得的可复用 Skill，而不是 test-time 额外计算。

### 4.15 上下文与 token 预算

Qwen3-VL-8B-Instruct 官方支持原生 256K context，但 4090 上实际瓶颈是 KV cache。目前管线使用：

```text
max_model_len = 16384
```

官方 EB-HAB planner 已包含 system prompt、约 70 个动作说明、10-shot examples、图像、任务和历史。规划估计：

| 调用 | 预计输入上下文 | 相对 direct planner 增量 |
|---|---:|---:|
| Direct VLMPlanner | 约 4K–8K | 基线 |
| + Active Skill | +250–500 | +5%–10% |
| + Sparse ledger | +200–400 | 总计约 +8%–20% |
| Expected Branch | 确定性规则 | 0 token |
| Visual Evidence Call | 独立约 0.8K–1.5K multimodal tokens | 仅 evolution 阶段、按需触发 |
| Attribution/Patch | 独立约 1K–3K/cluster | 仅 evolution 阶段 |

推荐配置：

```text
max_model_len = 16384
max_completion_tokens = 1024
active_skill_budget <= 512
ledger_budget <= 384
planner_input_p95 < 12000
```

控制策略：

- history 保留最近 8–12 个 action-feedback，更早信息压缩到 ledger；
- 静态 10-shot/system/action list 使用 prefix caching；
- 每次请求记录 prompt/completion/image token usage；
- evidence branch 只对 ambiguous/mismatch event 触发；
- 不把 expected/evidence/attribution/patch 合并成同一个 prompt。

若 evidence 触发率为 20%–30%，预计 evolution 阶段总处理 tokens 比 direct rollout 高约 20%–50%；最终测试阶段主要只增加 500–900 planner tokens。

### 4.16 推荐代码结构

建议在 Skill-Pro-EmbodiedBench 官方对齐 planner 基础上新建独立 namespace，不继续耦合 Skill-Pro SMDP/PPO：

```text
vista_skill/
├── schemas.py                 # Predicate, Skill, TransitionEvent 数据结构
├── skill_init.py              # S0 与 environment schema 初始化
├── action_schema.py           # EB-HAB action → precondition/effect compiler
├── rollout_logger.py          # pre/post image、feedback、token、version 日志
├── belief_ledger.py           # 三值谓词、provenance、TTL
├── expected_transition.py     # 确定性 expected branch
├── evidence_extractor.py      # rule-first + visual fallback
├── mismatch.py                # predicate-level mismatch
├── attribution.py             # rule-first + constrained teacher
├── event_cluster.py           # recurrence 和 evidence aggregation
├── patcher.py                 # bounded field patch
├── candidate_gate.py          # static/transition/paired stages
├── lineage.py                 # version、accept/reject、rollback
└── evaluation/
    ├── task_metrics.py
    ├── attribution_metrics.py
    ├── cost_metrics.py
    └── fault_injection.py
```

推荐核心数据类：

```python
@dataclass
class PredicateEvidence:
    predicate: str
    value: Literal["true", "false", "unknown"]
    confidence: float
    source: str
    evidence_id: str

@dataclass
class TransitionEvent:
    episode_id: str
    task_id: str
    step_id: int
    instruction: str
    action_id: int
    action_text: str
    skill_version: str
    pre_image: str
    post_image: str
    feedback: str
    expected_delta: list
    evidence_delta: list[PredicateEvidence]
    mismatches: list

@dataclass
class AttributionResult:
    target: Literal["belief_refresh", "skill_update", "abstain"]
    subreason: str
    field: Optional[str]
    evidence_ids: list[str]
    confidence: float
```

### 4.17 推荐执行伪代码

```python
skill = initialize_shared_skill()

for task in acquisition_stream:
    trajectory = rollout(frozen_executor, skill, task)

    for transition in trajectory.transitions:
        expected = expected_branch(
            ledger=transition.pre_ledger,
            skill=skill,
            action=transition.action,
            schema=fixed_action_schema,
        )

        evidence = evidence_branch(
            pre_ledger=transition.pre_ledger,
            action=transition.action,
            pre_image=transition.pre_image,
            post_image=transition.post_image,
            feedback=transition.feedback,
        )

        mismatch = compare(expected, evidence)
        if not mismatch:
            continue

        attribution = assign_credit(mismatch, evidence, skill)

        if attribution.target == "belief_refresh":
            update_episode_ledger(evidence)
        elif attribution.target == "abstain":
            record_audit_event(attribution)
        else:
            add_to_evidence_cluster(attribution, mismatch)

    for cluster in ready_clusters(min_independent_support=2):
        candidate = bounded_patch(skill, cluster)
        if staged_gate(parent=skill, candidate=candidate, dev_tasks=dev_split):
            skill = candidate

freeze(skill)
evaluate_on_final_test(skill)
```

---

## 5. Motivation 与 Method 图的绘制方案

### 5.1 Motivation Figure

采用横向三栏。

#### Panel (a)：Task Execution

展示：

```text
Instruction: Place all apples on the TV stand.
Action: Place apple_1 on the TV stand.
```

配动作前后图：

- 前：持有 `apple_1`，另有 `apple_2`；
- 后：`apple_1` 已放好，`apple_2` 仍未搬运。

展示错误 Skill termination：

```text
Stop after placing one target object.
```

#### Panel (b)：Prediction–Evidence Mismatch

橙色 Skill Prediction：

```text
on(apple_1, stand): false → true
task_complete: false → true
```

绿色 Visual Evidence：

```text
on(apple_1, stand) = true
on(apple_2, stand) = false
task_complete = false
```

红色突出：

```text
Skill predicts complete ≠ evidence supports incomplete
```

#### Panel (c)：Naive vs. VISTA

上方红色：

```text
trajectory failure → rewrite whole skill → negative transfer
```

下方绿色：

```text
action-level mismatch
→ skill/termination attribution
→ minimal patch
→ paired validation
```

灰色小分支：

```text
occluded apple_2 → unknown → abstain
```

### 5.2 Method Figure

横向主流程，上半部分是 evolution，下方单独画 frozen test。

```text
Generic S0 ───────────────┐
Fixed Action Schema ──────┼→ Frozen Planner → Action → Environment
                          │                       │
                          ▼                       ▼
                 Expected Transition     Evidence Transition
                          └───────╳───────┘
                          No Prediction Leakage
                                  │
                               Mismatch
                                  │
                         Hierarchical VTCA
                   ┌──────────────┼──────────────┐
              Belief Refresh  Skill Defect    Abstain
                                  │
                         Target → Skill Field
                                  │
                         Minimal Field Patch
                                  │
                          Paired Verification
                                  │
                     Accept ──────┴────── Reject
                       │
                       └────────→ Updated Skill S'
```

最下方：

```text
Frozen Final Skill S* + Frozen VLM + Current Observation → Action Plan
No attribution / no revision / no evolution teacher
```

颜色建议：

- 橙色：Skill / expected prediction；
- 绿色：independent evidence / accepted patch；
- 蓝色：episode belief refresh；
- 灰色：unknown / abstain / frozen component；
- 红色：mismatch / leakage barrier / rejected update。

主图不要再突出：

- 完整 scene graph；
- 四种长期 memory 同时更新；
- action model 在线学习；
- Skill-Pro PPO 结构；
- slow/fast/meta memory 细节。

---

## 6. 实验规划

### 6.1 论文研究问题

#### RQ1：Performance

> VISTA-Skill 是否在相同 executor、teacher、初始 Skill 和 evolution budget 下显著超过 EmbodiSkill？

#### RQ2：Attribution

> Evidence-decoupled action-level attribution 是否比 trajectory-level routing 更准确识别真正应该修改 Skill 的经验和字段？

#### RQ3：Reliability

> VISTA-Skill 是否提高 beneficial-update precision、降低 harmful-update rate 和 subgroup regression？

#### RQ4：Efficiency

> 在相同或更少 teacher calls/tokens 和 candidate-evaluation episodes 下，VISTA-Skill 是否取得更好的 performance–cost trade-off？

### 6.2 Benchmark 选择

#### Primary：EmbodiedBench-Habitat

原因：

- 当前复现与执行管线最成熟；
- 高层动作具有明确的 precondition/effect；
- 多实例、遮挡、空间关系、长程任务适合验证 VTCA；
- EmbodiSkill 有可比较的论文报数；
- 六个子集共 300 test tasks，成本可控。

#### Secondary：EmbodiedBench-Navigation

只有 EB-HAB 通过 Go/No-Go 后再运行。作用：

- 验证方法不只适用于 object rearrangement；
- 与 EmbodiSkill 的第二个视觉 benchmark 对齐；
- 只跑关键四行，避免扩张工作量。

#### 暂不作为主 benchmark

- EB-ALFRED：方法适配和动作空间成本较高，且不是当前 EmbodiSkill 视觉主表的最直接对比；
- ALFWorld：适合作为文本/无视觉消融，但不能证明视觉转移 credit；
- EB-Manipulation：低层连续/离散控制引入额外 perception/control confound。

### 6.3 数据划分

当前公开/复现管线中有 100 个 `train_validation` tasks。主协议建议固定：

```text
60 acquisition / 20 selection-dev / 20 frozen audit
```

要求：

- task-level split；
- 三个 evolution seeds 轮换 split；
- audit 对 updater 完全不可见；
- 六个官方 test subsets 仅在方案锁定后运行；
- 不从 final test failure 中手工修改 Skill；
- 所有 split、task IDs 和 hashes 随代码发布。

限制：EmbodiSkill 原论文使用 1000 training tasks，当前未公开。因此：

- 论文 reported number 单独列出；
- controlled EmbodiSkill\* 与 VISTA 使用同一公开 100-task 协议；
- 若作者后续释放 1000 tasks，再做一次 scale reproduction；
- 不把 100-task controlled result 称为 exact reproduction。

### 6.4 模型设置

#### 主 executor

```text
Qwen3-VL-8B-Instruct
```

部署：

- vLLM OpenAI-compatible server；
- 当前可行设置优先 fp8；
- `max_model_len=16384`；
- `max_completion_tokens≈1024`；
- 相同 image resolution、temperature、n-shot、action parser；
- 所有方法冻结 executor weights。

#### 主 evolution teacher

选择一个固定 teacher 完成全部主实验：

- 若优先贴近 EmbodiSkill 论文：使用其 GPT-5.2 配置或当前可访问的固定等价版本；
- 若成本优先：使用固定版本 Gemini Flash；
- 不在主实验同时展开两个 teacher；
- 第二 teacher 只做少量 robustness transfer。

要求：

- 所有 controlled methods 使用同一 teacher；
- 固定 prompt、temperature、candidate count、token budget；
- 记录准确模型版本和 API 日期；
- teacher 不参与 final test execution。

#### 32B scale executor

```text
Qwen3-VL-32B-Instruct
```

只在 8B 核心结果通过后运行。优先：

- RTX PRO 6000 96G 单卡 bf16；
- 可先用双 4090 + 官方 FP8 做 20–50 episode smoke；
- 双 4090 PCIe TP=2 不作为最终最可靠部署。

### 6.5 Baseline 阵容

#### 主性能表：六行

| 方法                          | 作用                                       | 优先级 |
| --------------------------- | ---------------------------------------- | --- |
| No Skill                    | frozen VLM 下界                            | 必须  |
| Static Shared Skill         | 隔离初始化 Skill 的贡献                          | 必须  |
| EmbodiSkill\* Native        | 主要原生 baseline                            | 必须  |
| EmbodiSkill\* + Common Gate | 最强受控 baseline                            | 必须  |
| VISTA w/o VTCA              | 相同 updater，所有失败直接进 trajectory reflection | 必须  |
| Full VISTA-Skill            | 完整 prediction/evidence/attribution/gate  | 必须  |

#### Reported reference block

单独灰色区块列：

- EmbodiSkill-GPT reported；
- EmbodiSkill-Gemini reported；
- 其他论文 reported baselines。

标记训练任务和协议不同，不参与 controlled significance claim。

#### Appendix/diagnostic

- Skill-Pro official-aligned no-skill/seed/full；
- Skill-Pro PPO 失败原因与 cost；
- Mem0/G-Memory 等已有数字；
- 可选 SkillOpt/HDSO-style gate。

Skill-Pro 不再写成 `sec/4_experiments.tex` 中的 principal baseline，当前实验章节需要后续改写。

### 6.6 公平性控制

主对比固定：

- executor checkpoint；
- initial Skill semantics；
- Skill token budget；
- action schema；
- acquisition tasks/trajectories；
- teacher model；
- teacher calls/generated tokens；
- candidate count；
- edit scope；
- validation episode budget；
- final test tasks；
- prompt/system/10-shot；
- image resolution；
- random seeds。

必须分开报告：

1. native method pipeline；
2. common-updater controlled pipeline。

否则无法区分提升来自 attribution 还是更强 updater/gate。

### 6.7 主指标

#### 第一优先级：性能

- Task Success；
- Task Progress；
- 六子集平均；
- per-subset score；
- worst-group/subgroup regression。

#### 第二优先级：执行质量

- Invalid Action Ratio；
- Premature Termination Rate；
- repeated/no-op action ratio；
- average steps/plans per successful episode。

#### 第三优先级：机制与更新可靠性

- target Macro-F1；
- field Macro-F1；
- beneficial-update precision；
- harmful-update rate；
- missed-beneficial-update rate；
- abstention precision/coverage；
- evidence predicate F1；
- ECE/Brier score；
- teacher calls/tokens；
- local evidence extractor tokens；
- candidate-evaluation episodes；
- wall-clock/GPU hours。

### 6.8 必做消融

控制为四个高价值消融：

1. **No decoupling**：Evidence Branch 故意看到 expected prediction；
2. **No routing**：所有 failure/mismatch 都进入 Skill update；
3. **No abstention**：强制 belief 或 Skill 二选一；
4. **No paired gate**：生成 patch 后直接写入。

可选消融：

- raw trajectory vs. sparse predicates；
- full rewrite vs. field patch；
- rule-only evidence vs. rule+visual；
- oracle evidence；
- empty vs. spec-initialized Skill。

当前不做或后置：

- full scene graph；
- persistent action-model evolution；
- coupled graph–skill update；
- slow consolidation/meta-memory；
- 大量 optimizer variants。

### 6.9 结构化 fault-injection/归因评测

建议构造最小 action-level diagnostic set。

#### Belief faults

- 删除一个 object instance；
- 合并 `apple_1/apple_2`；
- stale holding/on/open state；
- unknown 被错误写为 false。

#### Skill faults

- activation 错误；
- checklist/procedure 遗漏；
- effect 错误；
- premature termination；
- constraint/recovery 错误。

#### No-write events

- executor 跳过正确 Skill；
- occlusion/insufficient view；
- stochastic/no-op；
- ambiguous feedback。

#### Action-schema faults（diagnostic/P2）

- 删除 near precondition；
- 错写 holding constraint；
- 错写 pick/place effect。

该类在主性能系统中不在线学习，但可用于验证原始四类问题定义的扩展性。

### 6.10 统计协议

- success 使用同任务 paired evaluation；
- McNemar test 或 paired bootstrap；
- task 和 evolution seed 做 hierarchical bootstrap；
- 报告 95% confidence interval；
- final 300 tasks 不反复用于开发；
- 三个 evolution seeds；
- stochastic executor 设置下至少三个 rollout seeds；
- 每个 EB-HAB subset 仅 50 tasks，必须优先使用 paired statistics 而不是只报平均数。

建议最终性能目标：

```text
Full VISTA vs. strongest controlled EmbodiSkill*+Gate:
≥ +5 absolute success points and 95% CI > 0
```

小于 3 个点即使方向为正，也很难作为性能主导的 CVPR 主张。

### 6.11 分阶段实验路线

#### Phase 0：协议锁定与 token profiling

任务：

- 固定 EmbodiedBench commit/dataset hash；
- 固定 100-task split；
- 跑 50 episodes 记录 prompt/image/completion tokens；
- 固定 max context、history window、Skill budget；
- 确认 No Skill 数字与当前官方对齐管线一致。

输出：protocol manifest、token/cost profile。

Go 条件：无系统性 JSON/context/环境错误。

#### Phase 1：Controlled EmbodiSkill\*

任务：

- 移植 EmbodiSkill skill/lapse routing；
- 保留其 native body/appendix 逻辑；
- 记录所有与官方 ALFWorld 实现不同的 EB-HAB adaptation；
- 跑 Static、EmbodiSkill\* Native 和 EmbodiSkill\*+Common Gate。

输出：最主要受控 baseline。

Go 条件：结果稳定、三次 evolution seed 方差可接受。

#### Phase 2：Evidence Branch pilot

任务：

- 实现 TransitionRecord；
- rule-first evidence；
- 视觉 fallback；
- 标注/审计 200–300 events；
- 计算 predicate F1、coverage、ECE。

Go 条件：关键 goal predicates 的 selective accuracy 足以支持 attribution；低置信事件可以可靠 abstain。

#### Phase 3：VTCA MVP

任务：

- expected compiler；
- mismatch；
- 三路 routing；
- target→field；
- recurrence；
- bounded patch；
- small proxy gate。

只运行：

- EmbodiSkill\*+Gate；
- VISTA w/o VTCA；
- Full VISTA。

使用 60–120 个任务的 pilot，而不是立即跑完整主表。

建议 pilot Go 条件：

- Full 相对 strongest controlled baseline 约 `+8` absolute points，或表现出足够大的稳定趋势；
- target Macro-F1 比 trajectory routing 高约 10 点；
- harmful update 相对下降至少 30%；
- teacher token 不高于 EmbodiSkill\*。

这些是项目门槛，不是待写入论文的已有结果。

#### Phase 4：完整 8B 主实验

任务：

- 六行主表；
- 300 EB-HAB test tasks；
- 三个 evolution seeds；
- 四个核心消融；
- reliability/cost 表；
- performance–cost Pareto；
- qualitative case studies。

只有 Phase 3 Go 后执行。

#### Phase 5：Scale 与第二环境

任务：

- 32B 关键行；
- EB-Navigation 关键四行；
- 第二 teacher 小规模 transfer；
- final benchmark/diagnostic release 决策。

不再复制所有 8B 消融。

### 6.12 硬件规划

当前条件：

- 若干单卡 RTX 4090/4090D 24GB；
- 两个双卡节点；
- 最大可接受租用 RTX PRO 6000 96G。

推荐分配：

#### 单卡 4090 节点

- Qwen3-VL-8B vLLM；
- 独立 subset/seed rollout；
- evidence offline batching；
- 不使用 tensor parallel；
- 复用已缓存图像和 trajectory。

#### 双 4090 节点

8B 阶段优先：

- 两张卡分别跑独立 subset/seed；或
- 一张卡跑 planner server，一张卡批量跑 evidence extraction。

不建议 8B 使用 TP=2。

32B 阶段：

- 可用官方 FP8 + TP=2 做 20–50 episode smoke；
- 注意 PCIe、multimodal vLLM 和 Habitat 稳定性；
- smoke 失败不应阻塞核心论文。

#### RTX PRO 6000 96G

只用于：

- Qwen3-VL-32B bf16 最终关键行；
- 8B 核心结果明确成立后；
- 不用于开发；
- 不用于救 Skill-Pro PPO。

### 6.13 资源与耗时预估

以下为规划预算，应先用 50-episode profiling 校准。

| 阶段 | 主要工作 | GPU 预算 | 预计墙钟时间 |
|---|---|---:|---:|
| Phase 0 | 协议、50-ep profiling、日志 | 5–10 4090 GPUh | 1–2 天 |
| Phase 1 | EmbodiSkill\* port 与 baseline | 10–20 4090 GPUh | 3–5 天 |
| Phase 2 | Evidence extractor、审计集 | 10–20 4090 GPUh | 4–7 天 |
| Phase 3 | VTCA MVP、60–120 task pilot | 10–20 4090 GPUh | 4–7 天 |
| Phase 4 | 8B 主表、三 seed、消融 | 40–80 4090 GPUh | 4–7 天并行运行 |
| Phase 5a | 双 4090 32B FP8 smoke | 4–12 GPUh | 0.5–1 天 |
| Phase 5b | PRO 6000 32B bf16 关键行 | 15–30 96G GPUh | 1–3 天 |
| Optional | EB-VTCA benchmark 扩建 | 10–30 4090 GPUh | 1–2 周工程/标注 |

核心项目总量预估：

- 工程与实验并行：约 3–5 周；
- 本地 4090：约 75–150 GPUh，包括失败重跑余量；
- 正式 32B 租卡：约 15–30 GPUh；
- 若缓存和并行做得好，墙钟时间远低于 GPU-hour 总和；
- teacher/API 成本用 calls/tokens 记录，不用易变化的人民币单价写论文。

现有 Skill-Pro 实测可作为吞吐下界参考：

- 250 episodes + evolution：约 4–6 小时/单 4090；
- 300 episode no-skill eval：约 1 小时；
- Skill-Pro SMDP 300 episode：约 2–2.5 小时。

VISTA 不使用 per-step meta-controller 和 PPO logprob，目标是接近 direct/Static Skill 的 rollout 吞吐；额外 evidence extraction 尽量离线批处理。

### 6.14 Cost reporting

论文至少报告：

- executor calls；
- local evidence calls；
- teacher calls；
- input/output tokens；
- candidate count；
- candidate-evaluation episodes；
- 4090/96G GPU-hours；
- acquisition wall-clock；
- final test calls。

目标：

- teacher 总 token 不超过 EmbodiSkill\*，争取不超过其 1/3；
- test-time model call count 与 Static/EmbodiSkill frozen Skill 相同；
- performance–cost Pareto 优于 EmbodiSkill\*；
- 不在未实测前声称固定倍数成本优势。

---

## 7. Optional Benchmark 路线

### 7.1 何时需要

如果：

- attribution 明显更准；
- harmful update 明显下降；
- 但最终性能只提升 0–3 个点；

则可转为方法 + benchmark 混合论文，而不是继续堆大模型。

如果 attribution 本身也不成立，则不应扩建 benchmark 来掩盖方法失败。

### 7.2 EB-VTCA 最小版本

名称建议：

> **EB-VTCA: Action-Level Visual Transition Credit Assignment Benchmark**

规模：

- 1k–3k action-level transition events；
- pre/post RGB；
- action 与 environment feedback；
- expected transition；
- evidence predicates；
- ground-truth/injected update target；
- Skill field；
- confidence/unknown/occlusion 标记。

标签体系：

```text
belief / action semantics / skill / no-write
```

Skill field：

```text
activation / procedure / effect / termination / constraint
```

数据纪律：

- scene/task/object-disjoint split；
- simulator state 生成 controllable labels；
- 200–300 条人工双标；
- 报告 inter-annotator agreement；
- 不使用 final EB-HAB test trajectory 构造训练标签；
- 发布 generator、fault manifest 和 evaluation script。

指标：

- target/field Macro-F1；
- selective risk/coverage；
- ECE/Brier；
- benefit prediction AUROC；
- harmful-update rate；
- downstream task success。

### 7.3 Benchmark 的论文价值边界

有价值的版本是：

> 第一个系统评测“视觉动作转移应该写入哪种适应组件”的受控具身数据集，并证明该归因能预测下游 Skill update 是否有益。

无价值的版本是：

> 简单收集大量前后帧，再让 VLM 做变化描述。

Benchmark 必须与 heterogeneous update attribution 和 beneficial/harmful update 建立直接联系。

---

## 8. Go/No-Go 与项目分流

### 8.1 Go：完整 CVPR 方法论文

需要同时满足：

- Full 在 controlled EmbodiSkill\*+Gate 上有明确性能优势；
- attribution 显著更准；
- beneficial-update precision 提升；
- harmful-update rate 下降；
- cost 不高于或 Pareto 优于 EmbodiSkill；
- 结果在多实例/遮挡/长程子集上有一致解释。

### 8.2 部分 Go：方法 + Benchmark

适用：

- attribution/reliability 显著；
- final success gain 较小或受 8B ceiling 限制；
- fault-injection 和 benefit prediction 证据很强。

此时补强 EB-VTCA、32B 少量结果和 failure analysis。

### 8.3 No-Go：停止 VTCA 主标题

若出现：

- Evidence Branch 在关键谓词上不可靠；
- target attribution 不优于 EmbodiSkill/trajectory routing；
- filtered events 不能预测 beneficial update；
- gate 只减少更新数，却不降低 harmful update；
- Full 无法超过 Static/EmbodiSkill controlled baseline；

则不应继续追加大卡成本。

可降级为：

- VISTA-Guard：视觉证据门控 Skill update；
- state verification for embodied skill execution；
- Skill-Pro/EmbodiSkill reproducibility and diagnostic study；
- 小型 EB-VTCA benchmark，但前提是诊断信号本身可靠。

### 8.4 Coupled update 的边界

只有当 persistent action-model update 或 coupled graph–skill update 显著优于 skill-only 时，才允许重新加入：

```text
co-evolution / bidirectional update
```

否则它们只作为 future work，不进入标题和贡献。

---

## 9. 论文写作建议

### 9.1 推荐贡献表述

1. 定义 action-level visual transition credit assignment，研究哪些具身经验应进入长期程序化 Skill；
2. 提出 information-decoupled expected/evidence transition interface，减少 Skill prediction 自证；
3. 提出 target→field hierarchical attribution 与 evidence-gated bounded update；
4. 在 matched EmbodiSkill baselines 下验证性能、更新可靠性和成本；
5. 可选：发布 simulator-controlled VTCA diagnostic benchmark。

### 9.2 不建议的贡献表述

- 第一个自进化具身 Skill 方法；
- 第一个可靠 Skill optimizer；
- 第一个 memory credit assignment；
- graph–skill co-evolution，除非 coupled ablation 成立；
- 零额外成本，除非测试阶段和 evolution 总成本均已严格统计；
- exact EmbodiSkill reproduction，除非 1000 tasks 和官方代码释放。

### 9.3 实验章节需要修改的地方

当前 `sec/4_experiments.tex` 仍把 Skill-Pro-VLM 写成 principal baseline，后续需要改为：

- principal baseline：EmbodiSkill\*；
- strongest controlled baseline：EmbodiSkill\*+Common Gate；
- Skill-Pro：appendix diagnostic；
- reported vs. controlled 分表；
- RQ2 从“Skill-Pro semantic gradients”改为“trajectory-level/EmbodiSkill routing”；
- main table 先聚焦 EB-HAB，EB-Navigation 作为第二环境；
- 补充 initialization、split、token 和 cost protocol。

---

## 10. 近期执行清单

按顺序完成：

1. 冻结 EmbodiedBench、dataset、prompt、model 和 100-task split manifest；
2. 在 50 episodes 上记录真实 prompt/image/completion token 分布；
3. 定义共享 `S0`、Skill schema、token budget；
4. 实现 EmbodiSkill\* Native 与 EmbodiSkill\*+Common Gate；
5. 扩展 rollout logger，保存 action-level pre/post transition；
6. 实现 fixed action schema 与 expected compiler；
7. 实现 rule-first evidence extractor；
8. 标注/审计 200–300 transition events；
9. 实现三路 attribution 和五字段输出；
10. 实现 recurrence、bounded patch、staged gate；
11. 跑 60–120 task pilot，执行 Go/No-Go；
12. Go 后再跑 8B 主表、消融和 reliability/cost；
13. 8B 核心成立后再租 PRO 6000 跑 32B；
14. 最后决定是否扩展 EB-Navigation 和 EB-VTCA。

---

## 11. 关键参考

### 本地材料

- `context4agent/20260730-VISTA-Skill最终收敛方案-视觉状态转移信用分配与可靠Skill进化.md`
- `context4agent/20260730-VISTA-Skill问题定义与Skill-Pro对比核心Insights.md`
- `context4agent/Ju et al. - 2026 - EmbodiSkill Skill-Aware Reflection for Self-Evolving Embodied Agents.pdf`
- `context4agent/Mi et al. - 2026 - Skill-Pro Learning Reusable Skills from Experience via Non-Parametric PPO for LLM Agents.pdf`
- `../../Experiments/Skill-Pro-EmbodiedBench/docs/experiment_progress.md`
- `../../Experiments/Skill-Pro-EmbodiedBench/docs/experiment_results.md`
- `../../Experiments/Skill-Pro-EmbodiedBench/docs/migrate_to_big_gpu.md`

### 公开链接

- EmbodiedBench: <https://github.com/EmbodiedBench/EmbodiedBench>
- EmbodiSkill: <https://github.com/air-embodied-brain/EmbodiSkill>
- EmbodiSkill 1000 training tasks issue: <https://github.com/air-embodied-brain/EmbodiSkill/issues/2>
- Skill-Pro: <https://github.com/Miracle1207/Skill-Pro>
- SkillOpt: <https://github.com/microsoft/SkillOpt>
- HDSO: <https://arxiv.org/abs/2606.22330>
- SkillAudit: <https://arxiv.org/abs/2606.14239>
- Self-Supervised Skill Optimization: <https://arxiv.org/abs/2607.28777>
- HiMPO: <https://arxiv.org/abs/2606.16285>
- Qwen3-VL-8B-Instruct: <https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct>
- Qwen3-VL-32B-Instruct: <https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct>

---

## 12. 最终结论

当前最值得投入的版本不是一个完整、多 memory、全图联合进化系统，而是一个可以被独立定义和验证的视觉 Skill update attribution layer：

> **Skill 用结构化预期说明动作后应该发生什么，独立视觉证据说明当前真正有证据支持什么；系统先判断经验是否应由 Skill 承担，再限制修改字段，并用低成本配对验证阻止有害写入。**

论文能否成立，最终取决于四件事：

1. 在统一协议下是否真正超过 EmbodiSkill；
2. attribution 是否比 trajectory reflection 更准确；
3. accepted updates 是否更有益、更少有害；
4. 这一收益是否能在 4090 为主的成本下实现。

因此项目执行原则是：

> **先用最小系统验证 action-level visual credit signal，再扩性能、规模和 benchmark；不在核心假设成立前实现完整 graph/action-model co-evolution，也不为 Skill-Pro PPO 提前承担大卡成本。**
