# VISTA-Skill Phase 1 项目 Review

> Review 日期：2026-08-24
> Review 基线：commit `0b19569a2c8d06343615b5c21281fab7af9a207c`
> 项目阶段：Phase 1 closure（E1-E12）
> Review 范围：代码实现、Markdown/LaTeX 方案、实验日志、原始运行产物、测试和截至 Review 日期的公开相关工作

## 1. Executive Summary

VISTA-Skill 已完成一个较扎实的研究原型：真实 Habitat/AI2-THOR 环境、冻结 Qwen3-VL executor、视觉证据抽取、动作级转移归因、受限 Skill patch、配对候选门控、版本化产物和冻结评测已经形成端到端闭环。

但从科学证据看，项目目前仍处于**工程与诊断机制验证完成、论文主效果尚未成立**的阶段。Phase 1 最重要的事实是：

- stock EmbodiedBench 上没有任何持久更新被接受；
- 没有测得 Skill evolution 带来的性能恢复；
- attribution 的强结果主要来自与规则 taxonomy 匹配的合成集和少量注入故障；
- evidence branch 尚无 200-300 个自然事件上的独立准确率和校准结果；
- Full VISTA 的 teacher token 成本约为 trajectory baselines 的 20-26 倍；
- 当前方法只能修复已有的 Skill expectation，不能从空规则增长，也不能修复自然实验中暴露的 goal-coverage defect。

因此，项目不应继续按“已经证明更可靠、性能更高的通用 Skill evolution 方法”推进。更合理的定位是：

> 面向部分可观测具身执行的视觉动作转移级 Skill repair attribution，以及带字段真值、affected/protected subgroup 和反事实 repair/regression 审计的故障基准。

推荐采用经过修订的 benchmark-first 路线：先验证 evidence branch，再构建方法无关、预注册且具有独立审计集的 SkillFaultBench，最后才运行昂贵的 stock EmbodiedBench 主实验。当前最不应该做的是直接根据 selection 结果放宽 subgroup gate。

## 2. Review 方法与证据等级

本 Review 将证据分成三层：

1. **工程验证**：代码、测试、真实 simulator/model endpoint 和 artifact protocol 是否可运行。
2. **机制诊断验证**：在合成事件或注入故障上，组件是否执行了设计要求的行为。
3. **论文级受控证据**：多 seed、独立 split、公平预算和自然/外部分布上是否支持性能与可靠性主张。

这三类证据不能互相替代。例如，合成 attribution F1=1.0 可以证明规则实现与设计 taxonomy 一致，但不能作为自然事件 attribution accuracy 的估计；0 个 promoted harmful update 也不能在 promoted update 分母为 0 时证明 gate 的安全性。

## 3. 当前方案的实际实现

### 3.1 P0 数据流

当前实现以 [implementation.md](implementation.md) 和 [`vista_skill/`](../vista_skill/) 为准，而不是 LaTeX 中尚未更新的四记忆版本：

```text
固定 action schema + 五字段 Skill
              |
              v
动作前编译 Skill expected transition
              |
              v
执行 primitive action
              |
              v
动作后从 pre/post RGB 与公开反馈独立提取 evidence transition
              |
              v
构造 predicate-level mismatch
              |
              v
rule-first routing: belief_refresh / skill_update / abstain
              |
              v
跨独立 episode 的同字段 recurrence
              |
              v
单字段 bounded patch
              |
              v
static -> cached transition -> paired proxy -> paired finalist gate
              |
              v
接受并立即晋升，或拒绝并记录 lineage
              |
              v
冻结 Skill 后进行无 teacher 评测
```

关键实现位置：

- [`VistaSkillEngine.prepare/process_prepared`](../vista_skill/pipeline.py)：保证 expected branch 在动作前产生，evidence branch 只在动作后读取图像和公开反馈。
- [`CreditAssigner`](../vista_skill/attribution.py)：先处理证据不足、execution lapse、stochastic no-op、belief conflict，再将唯一的 Skill-sourced field contradiction 路由到 `skill_update`。
- [`CandidateGate`](../vista_skill/evolution.py)：执行 static、transition consistency、paired bootstrap LCB 和 protected-subgroup 检查。
- [`update_audit.py`](../vista_skill/update_audit.py)：对所有静态有效候选保存 parent/candidate snapshot，并在独立 audit role 上进行事后配对审计。
- [`configs/vista_p0.json`](../configs/vista_p0.json)：固定 60 acquisition / 20 selection / 20 audit、3 个 evolution seeds、proxy=10、finalist=30。
- [`configs/methods.json`](../configs/methods.json)：记录六个受控方法及需要匹配的实验因素。

### 3.2 已实现的能力

- 五字段结构化 Skill、typed termination policy 和 compiled prediction rules。
- 稀疏三值 predicate ledger、实例标识、置信度和 provenance。
- expected/evidence 信息隔离以及 fail-closed 的 unknown/uncovered 处理。
- `belief_refresh / skill_update / abstain` 三路 attribution。
- 跨独立 episode recurrence 和唯一 evidence ID 检查。
- 单字段 exact-target patch，同时更新文本与 compiled view。
- static、transition replay、paired proxy/finalist 和 subgroup regression gate。
- append-only lineage、digest-checked Skill artifact、split rotation 和 frozen evaluation。
- No Skill、Static Skill、EmbodiSkill* Native、EmbodiSkill*+Common Gate、VISTA w/o VTCA、Full VISTA 六个实验臂。

### 3.3 未实现或明确不属于 P0 的能力

- persistent action-model update；配置中该能力为 disabled。
- 完整 scene graph 和 graph-skill co-evolution。
- 从 empty/minimal Skill 中发现新 expectation 的 rule growth。
- goal grounding 或 goal-coverage model 的修复路径。
- no-decoupling、no-abstain 等论文核心消融。
- release-grade ACQ=60、三 seed、完整官方 300-task 主实验。

### 3.4 工程验证状态

本次 Review 重新运行：

```bash
/root/miniconda3/envs/max_embench/bin/python -m pytest
```

结果为：

```text
203 passed in 0.98s
```

README 中的 167 tests 和 Phase 1 closure 中的 210 passing 均已过期，应统一为自动生成或 CI badge，避免人工维护计数。

## 4. Phase 1 实验结果

完整时间线见 [experiment_log_phase1.md](experiment_log_phase1.md)。

| 实验 | 关键结果 | 可以支持的结论 | 主要限制 |
|---|---|---|---|
| E1-E2/E6 | 真实 Habitat、AI2-THOR、Qwen/vLLM、gate rollout 和 artifact 流程跑通 | 实验装置和主要接口可运行 | 不能证明方法有效 |
| E3 合成归因 | trajectory target Macro-F1 `0.196`、field Macro-F1 `0`；VTCA 均为 `1.000` | rule-first 实现符合预设 taxonomy | 数据由同一 taxonomy 构造，不能估计自然事件准确率 |
| E4-E6 stock pilots | skilled methods 最终均冻结在 S0；0 accepted update；Full teacher tokens 高约 20-26 倍 | 自然 Skill fault 稀疏，门控和成本记录可工作 | 不能证明 VISTA 提升性能或优于 EmbodiSkill |
| E7-E9 注入故障 | VTCA 在 4/4 campaigns 定位正确字段；trajectory proposals 报告为 0/26；出现 candidate `1.0` vs parent `0.0` | 注入故障下动作级定位和定向修补可行 | campaign 数量很少，4/4 与 0/26 的统计单位不一致 |
| E8o/E8p selection | proxy mean `+0.195`、LCB `+0.0017`；finalist mean 约 `+0.21`、LCB `-0.0028` | global LCB 对稀疏 affected subgroup 可能低功效 | selection 增益未在独立 audit 中复现为安全候选 |
| E11 4B | 20 episodes 中 3 个自然 `skill_update`；abstain `91.6%` | 较弱 executor 略增冲突，但主要增加 lapse/不确定性 | 未达到预注册 density 和 abstain 标准 |
| T4 minimal Skill | Full 0 proposal；Full/no-VTCA/frozen 均 `0.400`，No Skill `0.550` | 当前系统是 repair-only；坏 Skill 可能主动误导 executor | 20-episode 结果噪声较大，不能估计普遍 effect size |

### 4.1 Stock EmbodiedBench

E5 的 EB-HAB base/20 结果为 No Skill `0.450`、S0/Full/EmbodiSkill* `0.500`；E6 复跑则为 No Skill `0.550`、所有 S0-based methods `0.450`。方向在两次小样本间反转，日志估计二项 95% CI 约为 `+/-0.22`。

这意味着 S0 与 No Skill 的小幅差异目前属于噪声级观察。由于所有可进化方法最终都冻结在相同 S0，方法间相同分数不是 VISTA 与 baseline 性能相等的可靠证据，而是“没有方法产生持久更新”的直接结果。

### 4.2 注入故障与归因

E3 的完美分数是重要的单元级机制检查，但具有明显的 taxonomy matching：case 是按照与 `CreditAssigner` 相同的 failure categories 构建的，而且 teacher 根本没有被调用。这项结果应表述为“规则覆盖了设计内故障”，而不是“真实视觉归因准确率达到 100%”。

E7-E9 更有价值，因为它们在真实 simulator rollout 中注入了结构化 Skill fault。VTCA 多次定位到被注入的字段，而 trajectory reflection 往往提出 procedure/effect patch。但目前只有 4 个被总结为正确字段的 campaign，且 baseline 以 proposal 数作为分母。论文需要改为统一的 event/campaign 统计单位、置信区间和 scene/object/task-disjoint 测试。

### 4.3 自然故障与方法边界

8B+S0 约每 20 acquisition episodes 只有 1 个 skill-attributable event；4B 将其提高到 3/20，但 abstain 达到 91.6%。更关键的是，自然 termination conflict 通常是：

> 当前 grounder 认为所有目标都已满足，但环境仍报告任务未完成。

这更像遗漏目标或错误 final-state grounding，而不是 termination Skill field 错误。当前五字段 patch 无法修复这种 goal-coverage defect。

minimal Skill 实验揭示了另一个结构限制：没有 compiled Skill expectation，就没有 Skill-sourced contradiction，Full VISTA 因而没有进入 `skill_update` 的入口。该方法应被称为 reliable repair，而不是 general skill growth/evolution。

### 4.4 成本与公平性

在 executor 支出近似匹配时，Full VISTA 的 teacher tokens 是 trajectory baselines 的约 20-26 倍。其根本原因是 Full 在 primitive-action 层调用视觉/evidence/attribution teacher，而 trajectory baseline 只在 episode 层反思。

这与设计文档中“匹配 teacher calls/generated tokens”以及“teacher token 不高于 EmbodiSkill*”的目标不一致。未来主实验必须同时报告：

1. equal episode/candidate budget；
2. equal teacher-token budget；
3. performance-cost Pareto；
4. tokens/calls/rollouts/wall-time/GPU-hours per accepted beneficial update。

## 5. E8p 原始审计纠偏

Phase 1 closure 将“0 harmful updates、0 false accepts”作为 gate safety 结论。但仓库的 [`update_reliability`](../vista_skill/metrics.py) 只在 `promoted=True` 的候选上计算 harmful rate；当 promoted update 数为 0 时，该指标按实现返回 0。

原始 [`update_audit.json`](../running/fault_repair_e8p_constraint/full/seed_0/update_audit.json) 显示：

- 7 个候选；
- 420 组 audit pairs，即 20 audit tasks x 3 rollout seeds x 7 candidates；
- 按当前 `epsilon=0` 定义，7 个候选全部被分类为 `harmful`；
- 7 个候选全部未 promoted。

代表性候选：

| Candidate | Overall delta | 最差 subgroup delta | Audit classification |
|---|---:|---:|---|
| `912598a84e72...` | `-0.0470` | `-0.2333` | harmful |
| `59975891618b...` | `+0.0174` | `-0.1252` | harmful |
| `24977d58506d...` | `-0.0618` | `-0.2093` | harmful |

由此得到两个结论：

1. 当前 gate 确实非常保守，但 E8p 中的拒绝可能是正确的，而不只是 global LCB 低功效造成的 false rejection。
2. selection 中的 subgroup-local benefit 没有自动转化为独立 audit 中的安全更新；直接改成 `affected-subgroup LCB > 0` 缺乏证据。

未来 subgroup gate 必须满足：

- affected/protected group 由故障语义在 rollout 前定义，不能根据 observed delta 事后挑选；
- gate-selection、threshold development 和 final audit 使用独立数据；
- protected group 使用预注册 non-inferiority margin 和 simultaneous confidence bounds；
- subgroup 采用人可解释语义标签，而不是仅使用 opaque hash；
- 报告 update coverage，避免 reject-all 获得表面上的 0 harmful rate；
- `epsilon` 使用具有任务意义的阈值，例如 0.03-0.05，而不是把任意随机负差都视为 harm。

## 6. 当前论文主张的证据状态

### 6.1 已有较强支持

- 真实 simulator、Qwen endpoint、版本化 artifact 和配对审计链路可以运行。
- expected/evidence 信息隔离在接口和数据结构层面得到了较完整的实现。
- rule-first VTCA 在匹配 taxonomy 的合成诊断中准确。
- gate 能拒绝 malformed、inert、未修复目标和 audit-harmful candidates。
- 自然 Skill fault 在 8B+S0 下非常稀疏。
- 当前系统无法从 empty/minimal rule set 增长。
- 动作级视觉 evidence/attribution 的 teacher 成本显著高于 trajectory reflection。

### 6.2 尚未得到支持

- VISTA 提高 stock EB-HAB 或 EB-NAV 最终性能。
- VISTA 的 beneficial-update precision 优于受控 baselines。
- VISTA 降低 harmful-update rate，而不仅是拒绝所有更新。
- persistent update 已经在独立 audit 中被证实有益并接受。
- evidence predicate F1、ECE、Brier 或 selective accuracy 足够可靠。
- 自然事件上的 target/field attribution accuracy。
- ACQ=60、三 evolution seeds 的 release-grade 受控结果。
- 原计划要求的 performance-cost Pareto 优势。

## 7. Markdown、LaTeX 与代码一致性

最新 Markdown [20260806 方案](../context4agent/markdown/20260806-VISTA-Skill最新方案-证据解耦视觉转移信用分配.md) 已明确把 P0 收缩为三路 routing、固定 action schema 和 sparse predicate ledger，整体上与实现相符。

LaTeX 则仍存在严重漂移：

- [`3_method.tex`](../context4agent/latex/sec/3_method.tex) 仍声明 episode graph、persistent action-effect memory、Skill pool 和 control state 四类 adaptive state，并保留 action-model update。
- [`4_experiments.tex`](../context4agent/latex/sec/4_experiments.tex) 的 quantitative results 为空，主表中的部分数字不是当前 VISTA 受控结果，仍包含尚未完成的 Skill-Pro-VLM 和 graph-related ablations。
- [`0_abstract.tex`](../context4agent/latex/sec/0_abstract.tex) 与 [`5_conclusion.tex`](../context4agent/latex/sec/5_conclusion.tex) 仍保留 `X/Y/Z` 或 `[TASK RESULTS]` 性能提升占位主张。
- related work 尚未覆盖 2026 年 4-8 月快速出现的 skill benchmark、step-level attribution 和 harmful-skill 分析工作。

在新的受控结果出现前，LaTeX 应将性能、beneficial precision 和 harmful reduction 全部写成研究问题或待验证假设，不能使用完成时主张。

## 8. 最新相关工作与项目定位

截至 2026-08-24，skill evolution 和 evaluation 已明显变得拥挤：

- [EmbodiSkill](https://arxiv.org/abs/2605.10332)：已提出 skill defect 与 execution lapse routing，是最接近的具身 baseline。
- [SkillAdaptor](https://arxiv.org/abs/2606.01311)：已提出 step-level failure attribution 和定向 Skill update，削弱了单独以“step-level attribution”为 novelty 的空间。
- [SkillsBench](https://arxiv.org/abs/2602.12670)：86 个任务、7,308 条轨迹；curated Skills 平均提升 16.2pp，但 84 个有效任务中 16 个出现负增益，自生成 Skill 平均无收益。
- [Rethinking Self-Evolving Agent Skills](https://arxiv.org/abs/2608.02636)：388 个 candidates 中只有 55 个成为 byte-distinct validation best；说明进化更像稀疏的 validation-filtered search。
- [Agent Skills Can Be Harmful](https://arxiv.org/abs/2608.11888)：通过 differential analysis 总结 307 个 skill-induced failures，直接覆盖“Skill 会产生功能和成本回退”的论点。
- [Long-Horizon Agent Trajectory Attribution](https://arxiv.org/abs/2608.06909)：发布 1,300+ 条带 attribution component/chain 标注的轨迹，限制了广义“首个 attribution benchmark”主张。
- [SkillLearnBench](https://arxiv.org/abs/2604.20087)、[SkillFlow](https://arxiv.org/abs/2604.17308)、[EvoAgentBench](https://arxiv.org/abs/2607.05202) 和 [SEAGym](https://arxiv.org/abs/2606.17546)：已分别覆盖 continual skill learning、lifelong skill evolution、ability transfer 和 train/validation/test/replay/cost protocol。
- [SkillGen](https://arxiv.org/abs/2605.10999) 和 [Self-Supervised Skill Optimization](https://arxiv.org/abs/2607.28777)：使用 paired intervention 或 unlabeled comparative validation 验证 candidate utility。
- [HiMPO](https://arxiv.org/abs/2606.16285)：已将 less-entangled credit assignment 用于 long-horizon memory writing。

因此不应声称：

- 第一个可靠 Skill optimizer；
- 第一个 self-evolving Skill benchmark；
- 第一个 step-level 或 memory credit assignment；
- 第一个发现 Skill 可能有害的工作。

SkillFaultBench 仍可能具有差异化，但必须明确限定为：

- embodied and visual；
- primitive-action transition based；
- update-target 与 Skill-field ground truth；
- partial observability、occlusion、identity 和 no-op controls；
- parent/candidate counterfactual repair 与 regression measurement；
- 语义预定义 affected/protected subgroups；
- scene/object/task-disjoint transfer。

## 9. 对当前三个研究计划的评价

### 9.1 Plan A：Benchmark-first

**结论：方向合理，但当前 gate-first 表述需要修改。**

优点：

- 直接解决 stock EB 中 fault sparsity、subgroup invisibility 和 attribution ground truth 缺失的问题；
- 能复用已经实现的 injection、paired evaluator、snapshot 和 audit infrastructure；
- 与 Phase 1 真实暴露出的限制一致。

风险：

- 如果 benchmark 的 fault taxonomy、evidence predicates 和 affected groups 都由现有规则倒推，会重现 E3 的 matched-taxonomy circularity；
- 如果先看 E8p 再定义 gate 和 subgroup，容易产生 benchmark/gate co-adaptation；
- 广义 skill benchmark 已经拥挤，必须保持具身视觉和 fault-level counterfactual 的明确边界。

修订建议：先完成 evidence Go/No-Go 和 benchmark calibration；gate semantics 作为要比较的实验变量，而不是未经验证的固定“修复”。

### 9.2 Plan B：Method-first

**结论：现阶段不合理。**

rule growth 和 goal-model repair 分别引入新的触发条件、归因目标、patch 表示和评测真值。它们会同时扩大方法与 benchmark 的不确定性，而 evidence branch 本身尚未被独立验证。只有 repair-only benchmark 通过后，才应选择其中一个作为 P1 扩展。

### 9.3 Plan C：Stock-EB regime sweep

**结论：优先级低。**

4B executor 和 minimal Skill 两个预注册 rescue 已分别因高 abstention 和无 expectation 入口失败。继续扩大 4B/8B x S0/minimal 的网格，较可能得到更多 fault-sparsity 或 capability-lapse 结果，而不是验证 VTCA 主张。

## 10. 推荐的未来实验规划

### 10.1 P2-0：锁定研究主张与离线 gate 分析

**研究问题**：E8p 的拒绝主要是 global LCB 低功效，还是候选确实不安全？

**实验**：

- 使用已有 7 candidate snapshots 和 selection/audit rollouts；
- 比较 global LCB、预定义 affected-group LCB + protected non-inferiority、hierarchical/shrinkage gate；
- 不用 E8p audit 选择最终 threshold，只用于提出假设和估算 variance/effect size；
- 在新 benchmark 数据生成前预注册 group definition、margin 和统计检验。

**输出**：gate protocol、estimand 定义和 power analysis，而不是新的性能主张。

### 10.2 P2-1：Evidence Branch Go/No-Go

**样本**：300 个自然 primitive-action events，按 action type、成功/失败、遮挡、identity ambiguity、no-op 和 goal predicate 分层；至少 20% 双人独立标注并仲裁。

**实验臂**：

- rule/public-feedback evidence；
- 当前 visual evidence provider；
- oracle evidence；
- 可控 noisy evidence。

**指标**：predicate-transition Macro-F1、contradiction precision/recall、coverage、ECE、Brier、selective risk curve、inter-annotator agreement。

**建议 Go 条件**：

- contradiction precision >= 0.90；
- selective coverage >= 0.50；
- predicate-transition Macro-F1 >= 0.80；
- ECE <= 0.10；
- 低置信事件能以高 precision 进入 abstain。

若该阶段失败，应停止 VTCA 主标题，转向 visual state verification/VISTA-Guard，而不是继续扩大进化实验。

### 10.3 P2-2：SkillFaultBench v0

**Repairable fault families**：

1. procedure ordering/omission；
2. effect polarity/target；
3. constraint applicability；
4. termination quantifier/condition。

**Out-of-scope controls**：

- goal-coverage/grounding defect；
- missing-rule/minimal Skill；
- execution lapse；
- occlusion/insufficient evidence；
- stochastic/no-op；
- clean Skill。

goal-coverage 和 missing-rule 在新增对应机制前应测试正确 abstention，而不是计入 repair recovery。

**规模**：4 families x 2 severity levels x (20 affected + 20 protected tasks) = 320 fault-task coordinates；另加至少 80 个 control coordinates。使用 scene/object/task-disjoint acquisition、selection 和 audit split，并在 paired calibration 中使用 3 个 rollout seeds。

**Fault admission criteria**：

- vocabulary-aligned；
- cached-transition replay 可验证；
- affected group 由 fault semantics 预定义；
- faulty-vs-correct Skill 的 affected effect 95% LCB >= 0.10；
- protected group 的 non-inferiority lower bound > -0.05；
- 不允许使用同一场景/对象实例同时进行 calibration 与 audit。

### 10.4 P2-3：Attribution 主实验

**实验臂**：

1. trajectory/unconditional reflection；
2. EmbodiSkill* trajectory routing；
3. Full VTCA；
4. VTCA without decoupling；
5. VTCA without abstention；
6. Oracle-evidence VTCA。

**指标**：target Macro-F1、field Macro-F1、abstention precision/recall、clean false-update rate、per-family confusion matrix 和 calibration。

**Go 条件**：Full VTCA 相对 strongest trajectory baseline 的 target/field Macro-F1 至少提高 0.10，task/campaign-level bootstrap 95% CI 不跨 0；clean/no-fault false `skill_update` rate 低于 5%。

必须统一统计单位。不能继续将 Full 的 campaign accuracy 与 baseline 的 proposal count 直接比较。

### 10.5 P2-4：Repair 与 Gate 主实验

**实验臂**：

1. corrupted frozen Skill；
2. EmbodiSkill* Native；
3. EmbodiSkill* + Common Gate；
4. VISTA w/o VTCA + Common Gate；
5. Full VISTA；
6. Oracle-evidence/attribution VISTA，作为机制上限。

**协议**：每个 evolution seed 使用固定 60 acquisition / 20 selection / 20 audit rotation，共 3 seeds。executor、teacher、initial faulty Skill、candidate/edit budget、rollout seeds 和 final evaluation 完全匹配。

**两个预算视图**：

- equal candidate/evaluation budget；
- equal teacher-token budget。

**主要 estimands**：

- affected-subgroup recovery；
- global population utility；
- protected-group non-inferiority；
- beneficial-update precision；
- harmful-update rate；
- missed-beneficial-update rate；
- cost per accepted beneficial update。

**建议 Go 条件**：

- Full 相对 corrupted frozen 的 affected recovery 95% CI > 0；
- Full 相对 strongest controlled baseline 的 recovery delta 95% CI > 0；
- protected-group lower bound > -0.05；
- 至少审计 30 个 promoted updates；若 30 个中 0 harmful，95% rule-of-three upper bound 才约为 10%；
- equal-token budget 下 Full 仍位于 performance-cost Pareto frontier。

### 10.6 P2-5：Stock EB-HAB 外部有效性

只有 P2-4 Go 后才运行六个官方 subsets、共 300 tasks。对每个 evolving method 使用 3 个 independently evolved frozen Skill artifacts，并使用 task-paired 统计。

该阶段的主要目标应是：

- 证明 benchmark 上获得的 repair 不破坏 stock performance；
- Full 相对 S0 的预注册 non-inferiority margin 建议为 `-3pp`；
- superiority 作为 secondary outcome，而不是强行要求在 fault-sparse stock split 上产生大增益；
- 报告 per-subset、worst group、steps、invalid actions、premature termination 和全部成本。

### 10.7 延后项目

以下项目应在 8B repair claim 成立后再启动：

- EB-NAV 剩余 subsets；
- EB-ALF adapter；
- 32B scale experiment；
- rule growth；
- goal-model repair；
- persistent action-model update；
- graph-skill co-evolution。

## 11. 推荐的论文路线

### 11.1 推荐主张

> VISTA-Skill studies whether action-level visual transition evidence warrants repairing a persistent embodied Skill, and SkillFaultBench measures attribution, counterfactual repair, and subgroup regression under partial observability.

论文结构建议：

1. stock EmbodiedBench 负结果与 harmful Skill 作为 motivation；
2. evidence-decoupled visual transition attribution；
3. SkillFaultBench 的 fault/control/subgroup protocol；
4. attribution 与 reliable repair 主表；
5. stock EB-HAB 作为 external-validity/non-regression 表；
6. failure boundary：goal coverage、missing rule、cost。

### 11.2 不应使用的主张

- VISTA 已经提高 stock EmbodiedBench 性能；
- gate 已被证明能产生高 beneficial precision；
- zero harmful updates 证明了系统安全；
- action-level attribution 在自然事件上达到 100%；
- 当前系统能完成通用 Skill evolution 或 rule discovery；
- SkillFaultBench 是首个 self-evolution benchmark。

## 12. 最终结论

Phase 1 的价值不在于已经证明 VISTA-Skill 胜出，而在于它比较清楚地暴露了这个研究问题真正困难的部分：

1. stock embodied benchmark 中可归因的 Skill fault 太稀疏；
2. 自然 conflict 很可能来自 goal coverage，而非已有 Skill field；
3. repair benefit 和 regression 都高度依赖任务 subgroup；
4. selection 上的局部增益不能替代独立 audit；
5. reject-all gate 可以产生表面上的 0 harmful rate；
6. 动作级视觉 teacher 成本可能抵消方法收益；
7. repair 与 growth 必须在问题定义中明确分开。

综合判断：**Plan A 可以继续，但应改成 evidence-first、benchmark-calibrated、gate-as-hypothesis 的版本。** 在 P2-1 evidence audit 和 P2-2 benchmark calibration 通过前，不建议追加完整 ACQ=60 主表、32B、EB-ALF/NAV 扩展或 rule-growth 开发。
