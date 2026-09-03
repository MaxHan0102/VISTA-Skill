# VISTA-Skill Phase 4 研究规划

> 记录时间：2026-08-27 18:19:42 CST (+0800)
>
> 定位：本文件是 Phase 4 开始前的向前研究规划，汇总 2026-08-27 围绕 EmbodiSkill 与
> VISTA-Skill 的讨论结论。它不覆盖 Phase 1--3 的历史实验日志，也不把尚未运行的方案写成
> 已验证结论。正式实验开始前，仍需把具体数据、阈值、seed 和停止条件冻结为新的预注册协议。

## 1. 本轮讨论回答了什么

本轮围绕以下问题展开：

1. 最新 VISTA-Skill 的方法 pipeline 是什么；
2. EmbodiSkill 在 EB-HAB/EB-NAV 上的性能提升应如何解释；
3. EmbodiSkill 与 VISTA-Skill 的 Skill 表示和演化机制有什么差异；
4. 为什么当前 VISTA-Skill 尚未复现 EmbodiSkill 报告的成功率提升；
5. VISTA-Skill 最有希望在哪些维度超过 EmbodiSkill，尤其是训练成本、样本效率和长程任务。

## 2. EmbodiSkill 对照的事实边界

### 2.1 论文报告

EmbodiSkill 的测试时执行可以概括为：冻结的 Qwen3-VL executor 读取演化完成的 Skill，
然后在 held-out EB-HAB/EB-NAV 任务上执行；GPT-5.2 或 Gemini-3-flash 只负责训练阶段的
reflection/revision，不更新 executor 权重，也不参加最终测试执行。

Qwen3-VL-8B-Instruct 的论文报数为：

| Benchmark | No memory | EmbodiSkill + GPT-5.2 | Delta |
|---|---:|---:|---:|
| EB-HAB average | 24.00% | 45.33% | +21.33 pp |
| EB-NAV average | 45.00% | 50.33% | +5.33 pp |

长程子集仍有明显空间：Qwen3-VL-8B + GPT-5.2 Skill 在 EB-HAB long-horizon 为 24%，
EB-NAV long-horizon 为 0%；Gemini teacher 对应为 16% 和 3%。

不能把 `No memory -> EmbodiSkill` 的全部差值严格归因于 Skill evolution。论文没有在
EB-HAB/NAV 主表提供 matched Static Skill，因此差值同时包含初始 Skill 注入、Skill evolution、
teacher、prompt 和训练任务的共同影响。论文在 ALFWorld 上才给出了 No Skill、Static Skill、
Skill-unaware 和 EmbodiSkill 的完整消融。

### 2.2 公开性与复现限制

- 论文声称 EB-HAB 和 EB-NAV 各使用 1000 个 training tasks、10 个 revision stages。
- 当前官方仓库 commit 为 `760126030eab1d33ec6a6f30988f0f1fb58df3a7`。
- 公开仓库当前只含 ALFWorld workflow，没有论文 EB-HAB/NAV 的 1000 个训练任务、adapter、
  初始/最终 Skill artifact 和完整评测脚本。
- 官方 issue #2 仍在询问上述 1000 个训练任务的生成和 split，当前没有回复。

因此后续必须严格区分：

- **EmbodiSkill reported**：论文原始报数，只作为外部参照；
- **EmbodiSkill\***：本仓库在公开 EB 数据和统一协议下的 controlled reimplementation；
- 在官方 EB 资产未公开前，不能把 EmbodiSkill\* 称为 exact reproduction。

## 3. 两种 Skill 及其演化机制

### 3.1 EmbodiSkill

论文中的 Skill 为：

```text
Skill = (Skill Body, Skill Appendix)
```

- Body 保存长期操作规则；
- Appendix 强调 executor 没有遵循的有效 body 规则；
- 成功轨迹产生 Discovery 或 Optimization；
- 失败轨迹产生 Skill Defect 或 Execution Lapse；
- 前三类修改 body，Execution Lapse 只修改 appendix。

当前公开 ALFWorld 源码把 body 实现为 Planning、Search、Execution、Failure Avoidance 四段
自然语言 manual，appendix 为锚定具体 body rule 的 execution notes。它适合开放式积累程序知识，
但没有 typed predicate 或 compiled transition rule。

### 3.2 VISTA-Skill P0

VISTA-Skill 使用五字段 Skill：

```text
activation / procedure / effect / termination / constraint
```

每个版本同时包含自然语言 statements、typed termination policy 和 compiled prediction rules。
executor 直接看到五字段自然语言 Skill 和 episode-local belief；compiled rules 用于 expected
transition、predicate provenance、credit assignment、repair replay 和 candidate gate。

核心 pipeline 为：

```text
当前 Skill + fixed action schema + local belief
        -> primitive action
        -> expected transition / isolated evidence transition
        -> predicate mismatch
        -> belief_refresh / skill_update / abstain
        -> independent recurrence
        -> one-field exact-target bounded patch
        -> transition repair + paired affected/protected validation
        -> promote or reject
        -> frozen evaluation
```

当前 VISTA 的实质定位应为 **reliable Skill repair**。它擅长修复已有 expectation，尚不具备
EmbodiSkill Discovery 那样从 empty/minimal Skill 开放式长出新规则的能力。

## 4. 当前 Results 支持什么

### 4.1 已有正证据

- 动作级 VTCA 在真实注入 fault campaigns 中 `4/4` 定位正确字段；trajectory reflection
  在对应累计 proposals 中 `0/26` 命中正确字段。
- Multihold constraint 与 pick-effect inversion 的 Qwen bounded repair 都达到 `10/10` 正确修复。
- 20 条 acquisition trajectories 已足以产生可独立审计的 beneficial repair candidate。
- Effect inversion 的 7 次 proposal attempts 中，4 次被 independent audit 判为 beneficial；
  最佳 candidate 的 affected delta 为 `+0.0821`、affected LCB 为 `+0.0051`，protected
  delta 接近 0。
- 短时 RGB history 在保留 feedback 时出现 executor progress 信号：`0.6389 -> 0.7597`，
  success 为 `0.6000 -> 0.6667`；但 30-task CI 跨 0，尚不能作为正式性能结论。

### 4.2 当前失败与限制

- Stock EB-HAB 小样本中没有 persistent update 被接受，Full 最终仍是初始 S0。
- Effect inversion 的 beneficial candidates 被当前在线 gate 全部拒绝，最终 Skill 没有升级。
- 当前 Full VISTA 的 method teacher tokens 是 trajectory baselines 的约 `20-26` 倍；成本优势
  当前不成立。
- Multihold 的 204 次 method calls 中，167 次来自 visual evidence，说明主要成本在逐动作视觉调用。
- Feedback-only evidence F1 约 `0.80--0.84`，images+feedback 为 `0.85--0.89`；视觉有增量，
  但当前不是主要可靠信息源。
- Images-only/temporal evidence coverage 和 contradiction recall 太低，不能授权 persistent update。
- 当前方法依赖已有 expectation；empty/minimal Skill 无法自动形成有效规则。
- 目前没有 release-grade 的 VISTA long-horizon frozen performance 结果。

## 5. 最新研究目标

总目标不是简单复制 EmbodiSkill，而是在以下可审计维度上超过它：

1. **样本效率**：更少 acquisition trajectories 获得第一个真实有益更新；
2. **经济成本**：不依赖闭源大模型，降低模型 calls、tokens、GPU hours 和 gate rollout；
3. **更新可靠性**：更高 correct-target/field attribution 和 beneficial-update precision；
4. **最终性能**：evolved Skill 相对 matched Static Skill 带来稳定提升；
5. **长程能力**：重点改善多实例遗漏、状态遗忘、重复动作、前置条件错误和提前终止；
6. **可复现性**：公开数据坐标、Skill lineage、成本、seed、candidate 和独立 audit。

## 6. 最有希望的突破方向

| 优先级 | 方向 | 依据当前结果的判断 |
|---:|---|---|
| 1 | Semantic affected/protected gate | 已有 beneficial candidate，当前主要死在接收端；最快形成真实 promotion |
| 2 | Selective-VLM VTCA | visual evidence 占绝大多数 method calls；规则/feedback 已覆盖大部分简单转移 |
| 3 | Episode-local long-horizon goal agenda | 结构化 belief 是 VISTA 相对自然语言 manual 的独特优势，且长程子集有明确空间 |
| 4 | 少轨迹 repair curve | 20 条轨迹已能产生 beneficial candidate，有机会在 sample efficiency 上显著领先 |
| 5 | Success-driven bounded discovery | 要全面超过 EmbodiSkill 必须补齐，但当前没有直接正证据，风险高于 repair 主线 |
| 6 | 纯视觉、长 prompt Meta-Skill | Phase 3B/3C 已 No-Go，不继续作为近期主线 |

## 7. 建议的新阶段路线

### Stage A：修复 candidate admission

目标：让已经可生成的 beneficial repair 真正进入 frozen Skill。

- patch 生成后、读取 outcome 前，根据被修改的 action/predicate/field 编译 affected task family；
- acceptance 使用 `affected LCB > 0`；
- protected tasks 要求 `LCB > -epsilon`；
- global mean 作为 secondary metric，不再让大量 unaffected zero-delta tasks 否决局部修复；
- 在新 held-out fault variants 上预注册验证，不能复用 effect-inversion 的 post-hoc 分组作为主结果。

Go 条件：至少一个 update 被在线接受，并在 independent audit 上同时满足 affected benefit 和
protected non-inferiority；否则不得进入官方性能主表。

### Stage B：构建低成本 Selective-VLM VTCA

每个动作的快速路径只使用 fixed schema、public feedback parser、ledger 和确定性路由；只有当视觉信息可能改变 `belief_refresh / skill_update / abstain` 决策时才调用 VLM。

```text
primitive transition
    -> deterministic feedback/schema evidence
    -> decision-sufficiency check
       -> sufficient: no model call
       -> ambiguous and update-relevant: one visual call
    -> recurrence cluster
    -> one patch call per cluster
```

优先实现：

- 按 action type 和 feedback authority 的 visual trigger；
- transition signature cache 和相同证据去重；
- cluster 形成前不调用 patch teacher；
- compiled correction 尽量由 evidence 确定，LLM 只生成自然语言 statement；
- gate 先做 cached replay，再做少量 affected/protected rollouts。

成本必须拆开报告：external API cost、local model calls/tokens、GPU hours、Habitat episodes，以及
cost per accepted beneficial update。VISTA 已经实现 external API cost=0，但尚未实现低总计算成本。

### Stage C：轨迹内长程状态与执行控制

在 episode-local ledger 之上增加可编译的 goal agenda：

```text
pending/satisfied goal predicates
object-instance bindings
action preconditions
subgoal completion certificates
bounded retry/recovery counters
termination certificate
```

它在轨迹内实时更新，但默认不写入长期 Skill。只有同类长程失败跨 episode recurrence 后，才允许
修改 procedure、termination 或 constraint。原始 RGB history 不直接作为长期记忆；它只用于更新
结构化状态或触发 bounded recovery。

长程评测除 success/progress 外，必须报告 premature termination、duplicate-subgoal rate、adjacent
repetition、invalid action、completed-goal regression、steps to recovery 和 action efficiency。

### Stage D：补齐 bounded discovery

只有 repair + gate 闭环成立后，再增加成功轨迹的 Discovery/Optimization 分支：

- 从成功轨迹提取重复出现、可验证的 missing obligation；
- 新规则必须能够编译成 observable prediction/test；
- 使用独立成功 episodes recurrence；
- 遵守 one-field/token budget 和 affected/protected gate；
- 与 EmbodiSkill 的开放式 manual growth 单独做 sample-efficiency 对照。

这一步是全面超过 EmbodiSkill 所必需的，但不能抢在当前 promotion 闭环之前扩张范围。

## 8. 核心实验设计

至少比较：No Skill、Static S0、EmbodiSkill\* Native、EmbodiSkill\*+Common Gate、VISTA w/o VTCA、
Full VISTA。所有方法匹配 executor、初始 Skill 语义、teacher checkpoint、acquisition coordinates、
Skill/token budget、candidate-evaluation budget 和 seeds。

训练效率用 learning curve，而不是只报最终单点：

```text
5 / 10 / 20 / 40 / 60 acquisition trajectories
```

主要指标：

- trajectories to first accepted-beneficial update；
- affected performance recovery AUC；
- frozen success/progress delta over Static S0；
- correct target/field Macro-F1；
- beneficial-update precision、harmful-update rate、missed-beneficial rate；
- calls/tokens/GPU-hours/rollouts per accepted-beneficial update；
- long-horizon success、progress 和 failure-mode breakdown。

只有完成 matched controlled comparison 后，才能将 VISTA 的少轨迹结果与 EmbodiSkill reported 的
1000-task 结果放在同一讨论中；不同数据、teacher 和 adapter 的绝对成功率不能直接宣称 superiority。

## 9. 当前最准确的论文定位

近期最强、最可证伪的论文主张应是：

> VISTA-Skill 利用动作级视觉转移信用分配和结构化进度状态，在不依赖闭源 evolution teacher 的
> 条件下，用更少的高价值经验实现比 trajectory-level reflection 更准确、更可靠的 Skill repair；
> 在受影响任务上获得收益，同时保护无关能力，并重点改善长程任务中的状态遗忘、重复执行和提前终止。

在观察到 online accepted-beneficial update 和独立 frozen performance boost 之前，不能声称：

- VISTA 已经全面超过 EmbodiSkill；
- VISTA 当前总训练成本更低；
- VISTA 已经提升 EB-HAB/NAV long-horizon；
- 0 harmful promotion 证明 gate 安全，因为当前 promotion 数仍为 0；
- fault-injection repair 等价于 stock benchmark 的通用 Skill growth。

## 10. 下一步决策顺序

1. 预注册并验证 semantic affected/protected gate；
2. 让至少一个 beneficial repair 完成 online promotion -> frozen audit 闭环；
3. 用 cached replay 开发 Selective-VLM trigger，显著压低当前逐动作调用成本；
4. 实现 episode-local goal agenda，并先做 matched long-horizon mechanism pilot；
5. 运行 5/10/20/40/60 trajectory learning curves；
6. 满足 Go 条件后再开放三个 evolution seeds、官方 long-horizon 和六子集评测；
7. 最后研究 success-driven bounded discovery 和 EB-NAV transfer。

## 11. 依据

- EmbodiSkill paper：`context4agent/PDF/Ju et al. - 2026 - EmbodiSkill Skill-Aware Reflection for Self-Evolving Embodied Agents.pdf`
- EmbodiSkill official repository：<https://github.com/air-embodied-brain/EmbodiSkill>
- 当前实现状态：[`implementation.md`](implementation.md)
- Phase 1：[`experiment_log_phase1.md`](experiment_log_phase1.md)
- Phase 2：[`experiment_log_phase2.md`](experiment_log_phase2.md)
- Phase 3：[`experiment_log_phase3.md`](experiment_log_phase3.md)
- 通俗结果摘要：[`experiment_log_plain_zh.md`](experiment_log_plain_zh.md)
