# VISTA-Skill 通俗实验进展

> 面向项目开发者的快速版本。最后更新：2026-08-25。
>
> 这里只记录“做了什么、什么有效、什么无效、现在卡在哪里”。完整数字和实验因果链见
> [Phase 1](experiment_log_phase1.md)、[Phase 2](experiment_log_phase2.md) 和
> [Phase 3](experiment_log_phase3.md)。

## 一句话现状

VISTA-Skill 已经可以从 interface-only S0 的自然交互中自主发现动作 effect 和可复用的
precondition/constraint，也能在部分结构化故障中生成正确修复；但尚无候选通过可靠的任务效用
验证，因此还没有完成“自主发现/改进 → 接受更新 → 稳定提高最终任务性能”的完整闭环。

## 我们到底在研究什么

具身 Agent 执行失败时，原因可能是：

- 当前没看清、认错物体或临时 belief 过期；
- Agent 没按 Skill 执行；
- 环境随机或动作没有真正生效；
- Skill 本身确实写错了。

如果把所有失败都写回长期 Skill，Skill 很快会被偶然错误污染。VISTA-Skill 要解决的问题是：

> VISTA-Skill 能否从自身的视觉具身交互中，低成本地发现、验证并积累真正提升任务成功率的
> 可执行规则？

这包含四种长期变化：发现原 Skill 不包含的规则、修复错误规则、优化正确但低效/脆弱的规则，以及
处理 executor 没有遵循正确规则的执行失误。无论哪一种，都只有在独立任务上确认“相关任务变好、
其他能力不退化”后才能接受。

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
- 当时的方法依赖已有 expectation，只会 repair；这个限制已在 Phase 5 的自然 Discovery 实现中解除。

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

## Phase 5：从自然交互自主发现规则

- 主实验从只有公开动作接口、五个字段都为空的 interface-only S0 开始，不再手工注入错误 Skill。
- 两个独立 episode 中自然出现的相同 action-bound 状态变化会被泛化成跨物体规则。已真实发现
  nav/pick/place 的 effect，也发现了“pick 前应确认目标物体 near”的 constraint。
- Discovery 补丁由证据唯一决定，不调用 patch teacher。EB-HAB 结构化 feedback 足够时采用
  event-triggered evidence，一次四步任务的方法 token 从 `6479` 降至 `1964`（下降 `69.7%`）。
- 第一次真实 10-task 配对 gate 中，constraint candidate 改善了两个任务、退化了一个任务；总体
  均值只增加 `0.00838`，affected-task bootstrap LCB 为 `-0.26973`，因此被正确拒绝，Skill 保持 S0。
- 这说明“初始 Skill 无法自主长出候选”的瓶颈已解决，但“候选被 executor 稳定采用并可靠提升成功
  率”仍未解决。当前不能把候选出现或局部轨迹改善写成最终性能提升。
- 已限制每轮最多评估一个新候选，优先 constraint/procedure，并避免同一 Discovery 随证据增长反复
  花费 paired rollout。诊断运行还可显式跳过 120-episode 独立 audit；正式 controlled run 不能跳过。
- 同一 constraint gate 已重复两次；三轮共 60 个 proxy rollout 的逐任务结果与完整动作轨迹完全一致，
  候选每次都因 affected-task LCB `-0.26973` 被拒绝。这确认 task 62/66 的改善和 task 69 的退化均可
  复现，但也确认这条规则没有继续进入 finalist/audit 的价值。
- 已修复成本日志只统计 acquisition、遗漏 gate/audit 的问题。首个完整成本诊断记录 executor 91 次
  调用、440,371 tokens，其中 paired proxy 占约 82%；另有 5 次 goal grounding、9,876 method tokens。
  每个 episode 的 usage 都写入 JSONL，求和与 manifest 完全一致。旧 E4/v2 的成本字段不能用于比较。
- E6 把 VTCA 可识别性、时序恢复规则和 sequential safe gate 真正接入执行。5 个 acquisition episode
  中 20/20 个 Skill update 都带通过的可识别性审计；时序 monitor 在 task 66 拦截一次原地重复 pick，
  之后轨迹恢复并成功；候选整体将该任务从失败变成功，但不能把收益单独归因于这一次拦截。task 69
  的每次失败之间已有一次“成功但无用”的导航，所以规则没有触发拦截，候选仍从成功退化为失败。
  10-task 均值增益仅 `+0.00257`，affected LCB 为 `-0.29048`，
  因此继续被拒绝。
- Sequential gate 按 4/6/8/10 task 查看，但这个正负收益近乎抵消的候选没有达到安全提前停止条件，
  仍完成全部 20 个 rollout。它验证了“不能确定就回退到原完整 gate”，本轮没有产生 sequential 节省。

## 截至目前，哪些 idea 真正 Work

1. 动作前 expected 与动作后 evidence 分离，避免模型拿预期反向“证明”自己。
2. 三值 belief：没看见应保持 unknown，不能直接当成 false。
3. 动作级 predicate mismatch 和完整 provenance。
4. Rule-first VTCA：先排除 evidence 不足、execution lapse、stochastic 和 identity/belief 问题。
5. 单字段、exact-target、同时修改文本与 compiled rule 的 bounded patch。
6. 独立 repair/regression audit：能发现“目标变好但其他任务变差”。
7. Append-only lineage、digest、固定 split/seed、冻结评测和中断恢复等实验基础设施。
8. Interface-only S0 的自然 effect/constraint Discovery，以及零 patch-teacher 的确定性候选生成。
9. Event-triggered evidence 在 feedback 完整的 EB-HAB 上显著降低方法 token，不移除跨环境视觉 fallback。
10. 可审计的 VTCA 可识别性条件，以及能在真实轨迹中拦截动作的 episode-local 时序规则状态机。

## 哪些 idea 当前不 Work

1. 仅靠 RGB 代替 simulator feedback，尤其用于 persistent Skill update。
2. 用更严格 Evidence Guard 过滤出可靠提升：目前只换来低 coverage。
3. Direct Qwen 独立决定 attribution：误更新过多。
4. Episode-level trajectory reflection 定位具体 compiled Skill field。
5. 当前 Candidate Gate 稳定接受并积累真正有益的自然发现或修复。
6. 三条通用文字 Meta-Skill 提高 8B 的 Skill 演化性能。
7. 已证明 VISTA-Skill 提升 EB-HAB/EB-NAV 最终性能——目前不能这样声称。
8. 仅要求失败后“做任意一次成功导航”就能消除错误重试；task 69 证明还必须约束导航带来新的目标证据。

## 当前最准确的项目结论

```text
已经打通：自然发现 effect/constraint 或发现已有 Skill 错误 → 定位字段 → 生成候选 → 可靠拒绝未证实更新

尚未打通：稳定发现更高价值规则 → 在线接受有益候选 → 升级 frozen Skill → 正式任务稳定提高性能
```

因此现在不应继续堆新的 Guard 或通用 prompt。下一步应回到原始 VISTA-Skill 主线，重点解决：

1. 在新的 development rotation 上发现“导航必须增加目标证据”的 observation-grounded search/procedure 规则；
2. 从自然失败中发现更直接影响执行的 procedure/optimization，而不只总结 primitive effect 或禁止原地重试；
3. 用多 evolution seed 证明至少一个候选能够被在线接受，并在独立 audit 和 frozen evaluation 中保持
   收益；
4. 闭环成立后扩展完整 EB-HAB 主实验，再推进 EB-NAV 与更大模型的受控比较。

## 阅读实验结果时必须记住

- 正确 patch 不等于任务性能一定提高。
- 0 harmful promoted 在 0 promoted 时不等于 gate 已被证明安全。
- Selection 上变好不等于 independent audit 上仍然变好。
- No-Go 不是实验失败：它明确告诉我们哪些额外模块不值得继续投入。
- 当前论文主线已转为 **visual transition credit assignment for low-cost reliable Skill evolution**；自然
  Discovery 已有机制证据，但通用 self-evolution 的最终性能证据仍未完成。

### 2026-09-07：P5.7 自然恢复规则的小规模实测

两台 Qwen3-VL-8B 服务已经实际用于并行实验。12 次自然采集找到 8 个独立回合的重复失败链，生成了 1 条受限 procedure 候选。随后 4 次配对机制检验中，对照成功 1/2、候选 0/2；虽然无新证据的重复抓取从 7 次降为 0 次，但候选因拦截上限提前结束，不能算效率提升。

复查发现 UNKNOWN 状态会让后续视觉查询停止，已修复并保持候选不变，另行预注册执行 4 次复查。视觉查询恢复，但目标仍保持 UNKNOWN，成功数仍是对照 1/2、候选 0/2。因此两轮均 No-Go，没有晋升，也没有打开后续评估集。共 20 回合、175 次模型调用、653,606 tokens（不含服务探针）；完整测试 295 项通过。

目前的瓶颈是：规则可以阻止重复动作，但缺少可靠的目标搜索和证据获取路径。下一步先在新登记的开发任务上检验目标导向搜索是否真的增加可用证据，再验证任务收益；不能通过放松证据标准或继续调这两个筛选任务来制造正例。任务使用的是旧布局上的新目标组合，不代表新场景泛化。详见 [E11 完整记录](experiment_log_phase5.md#e11--p57-bounded-target-evidence-recovery-pilot--2026-09-07)。
