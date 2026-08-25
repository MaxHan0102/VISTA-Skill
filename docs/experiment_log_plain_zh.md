# VISTA-Skill 通俗实验进展

> 面向项目开发者的快速版本。最后更新：2026-08-25。
>
> 这里只记录“做了什么、什么有效、什么无效、现在卡在哪里”。完整数字和实验因果链见
> [Phase 1](experiment_log_phase1.md)、[Phase 2](experiment_log_phase2.md) 和
> [Phase 3](experiment_log_phase3.md)。

## 一句话现状

VISTA-Skill 已经可以在部分结构化故障中**发现错误的 Skill 字段并生成正确修复**，也发现过经
独立审计确实有益的候选 Skill；但在线 Candidate Gate 仍会错杀有益候选，因此尚未完成
“自动接受修复 → 升级 frozen Skill → 稳定提高最终任务性能”的完整闭环。

## 我们到底在研究什么

具身 Agent 执行失败时，原因可能是：

- 当前没看清、认错物体或临时 belief 过期；
- Agent 没按 Skill 执行；
- 环境随机或动作没有真正生效；
- Skill 本身确实写错了。

如果把所有失败都写回长期 Skill，Skill 很快会被偶然错误污染。VISTA-Skill 要解决的问题是：

> 在部分可观测环境里，判断一次失败是否真的应该归咎于 Skill；如果是，只做最小修复，并且只有
> 在独立任务上确认“目标问题变好、其他能力不退化”后才接受更新。

当前方法不训练模型权重。executor、teacher 和 patch author 都使用冻结的
Qwen3-VL-8B-Instruct。

## 当前方法的通俗流程

```text
动作前：Skill 预测“应该发生什么”
              ↓
执行一个 primitive action
              ↓
动作后：RGB + public feedback 判断“实际发生什么”
              ↓
比较预期和现实
              ↓
刷新临时 belief / 修改 Skill / 证据不足先不改
              ↓
同类错误跨 episode 重复出现
              ↓
只修改一个 Skill 字段
              ↓
独立比较 parent 与 candidate
              ↓
接受并升级 Skill，或拒绝并保留记录
```

Skill 分为 activation、procedure、effect、constraint、termination 五个字段。修改时同时更新文字
和真正参与预测的 compiled rule，避免“文字改了，系统实际规则没改”。

## Phase 1：先证明整条链路能跑

### 做了什么

- 接通真实 Habitat、AI2-THOR、Qwen/vLLM、Skill prompt 和冻结评测。
- 实现动作前 expected transition、动作后 evidence、三路归因、recurrence、bounded patch、
  paired gate、lineage 和独立 update audit。
- 用合成事件、stock 小样本和注入 Skill fault 检查各模块。

### 有效的部分

- 整套工程链路能够端到端运行，实验 artifact 可以复现和审计。
- taxonomy 对齐的合成诊断中，动作级 VTCA 的 target/field Macro-F1 都是 `1.0`。
- 真实 simulator 的少量注入故障中，VTCA 能定位到正确字段，也能生成定向候选。
- gate 能拒绝 malformed、没有真正修复目标、或独立 audit 中有害的候选。

### 无效或暴露的问题

- Stock EB-HAB 小样本里没有任何持久更新被接受，所有方法最后仍是初始 S0 Skill。
- 小样本中 Static Skill 和 No Skill 的胜负方向会反转，不能声称最终性能提高。
- Full VISTA 的 action-level teacher 成本约为 trajectory baseline 的 20–26 倍。
- 当前方法依赖已有 expectation，只会 repair；empty/minimal Skill 无法自动长出新规则。

### 这意味着什么

Phase 1 证明的是“装置和机制能运行”，不是“VISTA 已经让 Agent 变强”。它也把项目定位从泛化的
Skill growth 收缩为更诚实的 **reliable Skill repair**。

## Phase 2：验证证据、归因、补丁和真实修复

### 2.1 Evidence

- Images + feedback 的 weak-gold F1 约为 `0.85–0.89`。
- Feedback-only 约为 `0.80–0.84`。
- Images-only 约为 `0.50–0.55`，尤其难判断 pick 后是否真的 holding。

结论：视觉有少量增量，但当前高质量 evidence 主要依赖 public simulator feedback；纯视觉还不足以
支撑可靠的长期 Skill 更新。

### 2.2 Attribution

- Rule-first VTCA：误更新少、字段定位准，但会漏掉一部分真正的 Skill 错误。
- Direct Qwen teacher：召回高，但会把大量 control 错判成 Skill update，字段也容易选错。
- Provenance-aware partition 在开发诊断上明显提高了召回并保持低误更新，是当前最值得继续验证的
  idea；但它经过 post-hoc 修订，尚不能当作独立主结果。
- EmbodiSkill-style episode reflection 在 constraint/effect 两类 fault 上都无法命中正确字段，最终
  0 proposal；失败发生在 gate 之前。

结论：**动作级 predicate provenance 是 VISTA 当前最有价值的部分**。只看整段轨迹，模型能知道
“好像该总结经验”，却经常不知道具体应改哪条规则。

### 2.3 Patch

- Multihold constraint 和 pick-effect inversion 两类 compiled fault 都达到 `10/10` 正确修复。
- 修复能精确命中目标 rule，并保留同字段其他规则。

结论：冻结 8B 已经具备生成这种 bounded repair 的能力，patch author 不是当前主要瓶颈。

### 2.4 Live fault campaigns

- Multihold：生成了多个看似合理的补丁，但独立 audit 显示整体或 protected tasks 会退化；0 accepted。
- Effect inversion：7 次 proposal 中有 4 次经独立 audit 属于 beneficial，其中一个候选在
  fault-relevant affected/protected 后验检查中也稳定有益。
- 但原在线 gate 把这些候选全部拒绝，最终 frozen Skill 仍是故障版本。

结论：项目第一次证明了“VISTA 能提出实际有益的 Skill repair”，同时也确认当前最大瓶颈已经移到
**接收端**：通用 selection task 会稀释稀疏 fault 的收益，当前 gate 统计功效不足且过于保守。

## Phase 3：尝试增强鲁棒性，三个方向均未通过 Go/No-Go

### Phase 3A：Evidence Guard

想法：当视觉和 feedback 冲突时，用更严格的规则过滤可疑 evidence。

- Work：false contradiction 降到 `0`，没有新增错误 Skill update。
- Not work：coverage 下降约 `14.85` 个百分点，整体收益的置信区间不排除 0。

含义：它主要通过“少判断、少更新”减少错误，没有证明系统更有用。不能把 reject-all 式安全当成
可靠性提升。

### Phase 3B：减少对 feedback 的依赖

想法：用两到三帧 RGB 历史代替或补充环境 feedback。

- Work：三帧历史能把 no-feedback executor 的平均 progress 从 `0.5913` 恢复到 `0.6347`，接近
  feedback/current 的 `0.6389`。
- Not work：重复动作负担超过门槛；作为 Skill evidence 时 coverage 只有 `5.44%`，contradiction
  recall 只有 `1.57%`，没有召回任何 Skill update。

含义：RGB 历史可能帮助短期执行，但当前 8B 无法仅靠它产生足够可靠、足够完整的长期写入证据。
近期仍应保留 feedback，同时禁止单一不可靠信号直接触发 persistent update。

### Phase 3C：少量通用 Meta-Skill

想法：用 Observe-and-Recover、Attribute-and-Scope、Patch-and-Test 三条短 Meta-Skill 帮助同一个
8B 更可靠地演化 Skill，不训练新模型。

- Work：原有 rule-first 能挡住 Meta teacher 的坏判断；patch 仍为两类 `10/10`；表面变换一致率
  `94.44%`。
- Not work：Meta attribution 过度保守。Synthetic target F1 从 `0.528` 降到 `0.432`，field F1
  从 `0.939` 降到 `0.296`；两组 natural Skill-update F1 也下降；matched core token 成本增加
  `43.1%`。

含义：给同一个 8B 增加更长的通用文字检查表，不会自动带来更好的归因；本版本只是稳定地更加
保守。Phase3C v1 不接入主方法。

## 截至目前，哪些 idea 真正 Work

1. 动作前 expected 与动作后 evidence 分离，避免模型拿预期反向“证明”自己。
2. 三值 belief：没看见应保持 unknown，不能直接当成 false。
3. 动作级 predicate mismatch 和完整 provenance。
4. Rule-first VTCA：先排除 evidence 不足、execution lapse、stochastic 和 identity/belief 问题。
5. 单字段、exact-target、同时修改文本与 compiled rule 的 bounded patch。
6. 独立 repair/regression audit：能发现“目标变好但其他任务变差”。
7. Append-only lineage、digest、固定 split/seed、冻结评测和中断恢复等实验基础设施。

## 哪些 idea 当前不 Work

1. 从 empty/minimal Skill 自动长出新规则。
2. 仅靠 RGB 代替 simulator feedback，尤其用于 persistent Skill update。
3. 用更严格 Evidence Guard 过滤出可靠提升：目前只换来低 coverage。
4. Direct Qwen 独立决定 attribution：误更新过多。
5. Episode-level trajectory reflection 定位具体 compiled Skill field。
6. 当前 Candidate Gate 稳定接受稀疏但真正有益的修复。
7. 三条通用文字 Meta-Skill 提高 8B 的 Skill 演化性能。
8. 已证明 VISTA-Skill 提升 EB-HAB/EB-NAV 最终性能——目前不能这样声称。

## 当前最准确的项目结论

```text
已经打通：发现部分 Skill 错误 → 定位字段 → 生成正确修复 → 独立验证候选是否有益

尚未打通：在线正确接受有益候选 → 升级 frozen Skill → 在正式任务上稳定提高性能
```

因此现在不应继续堆新的 Guard 或通用 prompt。下一步应回到原始 VISTA-Skill 主线，重点解决：

1. 在新 held-out fault 上预注册验证 provenance-aware attribution，提高真错误召回率而不增加误更新；
2. 重新设计 fault-aware affected/protected gate，让真正相关的任务信号不被通用任务池稀释；
3. 用多 fault、多 evolution seed 证明至少一个候选能够被在线接受，并在独立 audit 和 frozen evaluation
   中保持收益；
4. 只有上述闭环成立后，再做 EB-HAB/NAV 主性能表和更大模型扩展。

## 阅读实验结果时必须记住

- 正确 patch 不等于任务性能一定提高。
- 0 harmful promoted 在 0 promoted 时不等于 gate 已被证明安全。
- Selection 上变好不等于 independent audit 上仍然变好。
- No-Go 不是实验失败：它明确告诉我们哪些额外模块不值得继续投入。
- 当前最强的论文故事仍是 **visual transition credit assignment for reliable Skill repair**，不是已经完成的
  通用 Skill self-evolution。
