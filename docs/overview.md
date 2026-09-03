# VISTA-Skill Overview

项目远程地址：<https://github.com/MaxHan0102/VISTA-Skill>

VISTA-Skill（Visual transition credit assignment for reliable Skill evolution）研究的是：具身智能体在部分可观测环境中执行动作后，如何判断观察到的变化究竟意味着临时信念需要修复、长期 Skill 需要更新，还是证据不足而应当暂不写入。项目的目标不是把每次失败都总结成经验，而是让 Skill 从自身的视觉交互中低成本地发现、修复和优化可执行规则，并在独立任务上验证更新确实带来收益且不会损害已有能力。

## 项目目标

VISTA-Skill 面向 EmbodiedBench 的 EB-HAB 和 EB-NAV 环境，使用冻结的 Qwen3-VL-8B-Instruct 作为 executor 和主要的自进化模型，通过非侵入式 adapter 接入 stock EmbodiedBench，不修改 benchmark 的核心 planner。当前 Phase 5 的严格目标是：在匹配 executor、协议、初始 Skill、经验和评测预算的条件下，同时超过已发表的 EmbodiSkill 结果（EB-HAB 45.33%，EB-NAV 50.33%），并取得更好的性能—进化成本折中。

项目同时关注以下可靠性要求：

- 一次失败不能自动被当作 Skill 缺陷；失败也可能来自视觉误判、belief 过期、执行器没有遵循规则、随机 no-op 或环境反馈不足。
- 长期更新必须有跨独立 episode 的重复证据，并且只能修改被归因的一个 Skill 字段。
- 候选更新必须经过执行器采用性检查、配对任务效用验证和独立回归审计；无法证明有益的候选应被拒绝或保留为 pending。
- 评测阶段冻结 Skill 和方法状态，隔离历史污染，并完整记录成功率、进度、无效动作、更新可靠性、子群回归、调用次数、token、rollout、时间和 GPU 成本。

## 核心研究问题

1. **视觉转移应由谁负责？** 在动作前 Skill 预测的变化与动作后证据支持的变化不一致时，系统能否区分 belief 错误、Skill 缺陷、执行失误、随机失败和证据不足，而不是直接修改长期记忆？
2. **能否从自然交互中形成可复用规则？** 从 interface-only 的空结构化初始 Skill 出发，能否利用成功和失败动作中的重复前置条件、过程、效果和终止证据，发现跨对象实例、跨场景可复用的规则？
3. **如何让更新既精确又不退化？** 字段级候选能否在独立任务上改善受影响任务，同时满足受保护任务的非劣性约束，并被执行器真正采用？
4. **可靠进化的成本是否可控？** 能否通过公共反馈解析、不确定性/新颖性触发、批量证据、缓存复用和 event-triggered visual fallback，减少逐动作视觉/教师调用，在保持归因和更新可靠性的同时低于匹配的 trajectory-level 控制？

## 方法总体概览

VISTA-Skill 由一个局部、带来源的谓词 ledger 和一个五字段程序化 Skill 共同构成。Skill 的字段为 `activation`、`procedure`、`effect`、`constraint` 和 `termination`；文字视图供冻结 VLM 执行，结构化/compiled 视图供预测、归因、执行检查和审计使用。Phase-5 主设置从 interface-only S0 开始：公开动作接口仍属于环境契约，但初始 Skill 不预置任务策略、动作效果或恢复规则。

每个 primitive action 都经过以下闭环：

```text
动作前：固定 action schema + 当前 Skill
                 ↓
          Expected transition
                 │
          执行 primitive action
                 │
动作后：public feedback + pre/post RGB + episode ledger
                 ↓
      Evidence-supported transition
                 ↓
        typed mismatch comparison
                 ↓
 belief refresh / skill update / abstain
                 ↓
 recurring independent evidence
                 ↓
       one-field bounded patch
                 ↓
 semantic affected/protected paired gate
                 ↓
       promote candidate or reject it
```

具体而言：

1. **Expected branch 与 Evidence branch 解耦。** Expected branch 只读取动作参数、固定 action schema、当前 Skill 和 pre-action ledger，编译出动作后的预期谓词变化。Evidence branch 独立读取动作前后图像、结构化环境反馈和局部历史，禁止读取 expected transition、Skill prediction、归因结果或候选补丁，避免 prediction leakage 和自我验证。
2. **先构造证据，再做分层归因。** 系统在 typed predicate 空间中比较 expected 与 evidence-supported transition。`unknown` 或覆盖不足会产生 unsupported/abstain，而不是 contradiction。规则优先使用 action feedback 和 ledger grounding；只有公共反馈无法解析相关转移时，才触发视觉模型作为 fallback。
3. **采用三路主路由。** `belief_refresh` 只修复当前 episode 的实例状态；`skill_update` 只在高置信、跨独立 episode 重复否定同一规则时触发，并继续定位到五个 Skill 字段之一；`abstain` 保留 `insufficient_evidence`、`execution_lapse`、`stochastic_noop` 和 `ambiguous` 等内部原因，不污染 canonical Skill。
4. **覆盖四类 Skill 生命周期。** 系统不仅支持 repair，还要支持从 `SUPPORTED_UNEXPECTED` 转移中进行 discovery，修复已有错误规则，优化正确但低效或脆弱的规则，以及在 executor 没有遵循有效规则时提高其可执行性/显著性而不误改规则。
5. **以证据门控长期进化。** 事件先按 Skill、字段、mismatch 类型和任务模式聚类，并要求独立 episode、唯一 evidence ID 和足够置信度。更新采用 exact-target、单字段、受限 token 预算的 bounded patch，同时更新文字和 compiled 视图。候选随后经过静态检查、缓存转移验证、executor-adoption 检查、语义上 outcome-independent 的 affected/protected 配对选择，以及独立 audit；只有受影响任务的收益和受保护任务的非劣性都满足门槛，才晋升新 Skill 版本。
6. **用冻结和审计保证可复现。** 每个 evolution seed 都固定任务角色轮换、模型/服务栈、提示词、Skill/schema/config digest、温度、token 上限和 rollout seed。接受与拒绝都写入 append-only lineage；冻结评测不再调用 attribution、patch 或 evolution teacher，历史 official-test 暴露产生的 artifact 会被隔离。

## 当前实现与证据边界

现有实现已经打通了从动作级 expected/evidence 分离、typed mismatch、层级归因、recurrence、bounded patch 到 paired gate、freeze 和成本审计的工程闭环。受控 fault 诊断中，动作级归因命中目标字段的案例为 4/4，而 trajectory reflection 为 0/26；结构化 bounded repair 在审计案例中为 10/10。Phase 5 进一步实现了从空 S0 的自然 effect discovery 和 `pick` 前 `near(object)` 约束 discovery，并验证了 event-triggered evidence 可以在反馈完整的 EB-HAB 轨迹上显著降低方法 token。

这些结果证明的是机制可运行和候选可生成，不等于最终任务性能已经提升。首次自然约束候选在重复 paired proxy 中稳定出现了局部改善，但由于受影响任务的 bootstrap 下置信界仍为负，候选被正确拒绝，冻结 Skill 保持 interface-only S0。当前的主要瓶颈是让更直接的 procedure/optimization 规则被 executor 稳定采用，并以足够统计功效通过任务效用 gate；在此之前，项目不会宣称已经超过 EmbodiSkill 或已经稳定提升 EB-HAB/EB-NAV 总体成功率。

## 项目结构与文档入口

- `vista_skill/`：VTCA、evolution、controlled baselines 以及 EmbodiedBench adapters 的实现。
- [`docs/implementation.md`](implementation.md)：当前代码架构、信息隔离、不变量、CLI 和实验协议。
- [`docs/research_plan_phase5.md`](research_plan_phase5.md)：截至 2026-09-02 的研究目标、阶段门槛和执行顺序。
- [`docs/experiment_log_phase5.md`](experiment_log_phase5.md)：Phase 5 的可复现实验记录与当前结论。
- [`docs/evaluation_integrity.md`](evaluation_integrity.md)：数据边界、artifact 资格和公平比较规则。
- [`docs/experiment_log_plain_zh.md`](experiment_log_plain_zh.md)：面向开发者的中文实验进展摘要。

