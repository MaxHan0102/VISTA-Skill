---
tags:
  - AIGC
---

# VISTA-Skill 最终收敛方案：视觉状态转移信用分配与可靠 Skill 进化

> 本笔记备份 2026-07-27 至 2026-07-30 围绕 VISTA-Skill 的问题定义、Skill 表示、Method、Motivation/Method 图、Skill-Pro/EmbodiSkill/SkillOpt 源码调研、实验设计和论文定位最终收敛出的方案。
>
> **这里的“最终收敛”指当前最值得实现和写入论文的研究设计，不代表尚未完成的实验结论已经得到验证。** 文中会把“论文必须成立的核心”“推荐但可后置的增强模块”“已淘汰的旧方案”明确区分。

---

## 0. TL;DR

### 一句话研究问题

> **当具身 Agent 执行动作后的视觉证据与 Skill 预期效果不一致时，应该把这次失配归因给当前视觉信念、长期动作知识、可复用 Skill，还是不写入任何长期记忆？**

英文推荐表述：

> **Given an action-level mismatch under partial observability, which adaptive component—episodic belief, persistent action knowledge, reusable procedural skill, or no persistent memory—should absorb the discrepancy, and what independent visual evidence supports that assignment?**

推荐的问题名称：

> **Visual Transition Credit Assignment for Procedural Skill Evolution**
>
> 面向程序化 Skill 进化的视觉状态转移信用分配

### 一句话方法

> **VISTA-Skill 让 Skill 在结构化谓词空间中显式预测动作效果，同时从动作后的图像和环境反馈中独立提取证据支持的信念变化；系统先决定错误应该由哪种记忆承担，只有重复出现、证据充分的 Skill 缺陷才会触发字段级补丁，并经过配对验证后写入长期 Skill。**

### 一句话创新

> **现有方法主要研究“怎样从轨迹优化 Skill”；VISTA-Skill 研究更上游的问题——“这条经验究竟应不应该优化 Skill，以及如果要改，责任属于 Skill 的哪个字段”。**

### 与三条最强相关路线的关系

| 工作 | 它主要解决什么 | VISTA-Skill 仍然要解决什么 |
|---|---|---|
| **Skill-Pro** | 用 semantic gradient、候选生成和 Non-Parametric PPO Gate 优化结构化 Skill | 失败发生后，经验是否真的应该进入 Skill 优化器 |
| **EmbodiSkill** | 区分 Skill defect 与 execution lapse，并分别修改 Skill body / appendix | 在动作级视觉状态变化上，把责任进一步分给 belief、action model、skill 或 no update |
| **SkillOpt** | 把一个 Markdown Skill 当作外部可训练状态，通过受限 patch、验证集 gate、快慢更新稳定优化 | 为优化器提供独立、细粒度、可验证的视觉 credit signal，而不是仅靠整条文本轨迹反思 |

> [!important] 最有价值的 Insight
> VISTA-Skill 不是“Skill-Pro 加一个图”，也不是“EmbodiSkill 加一个视觉输入”，更不是“SkillOpt 换一个具身 benchmark”。它的核心是一个可单独定义、标注和评测的 **action-level update-target attribution layer**。

---

## 1. 方案演化：从“做一个系统”收敛到“定义一个问题”

### 1.1 最初版本

最初的论文主线大致是：

> 联合维护视觉交互图和程序化 Skill，并根据二者的失配实现 graph–skill co-evolution。

这个版本包含不少合理模块：

- episode-level visual interaction graph；
- persistent action-effect graph；
- graph-grounded Skill；
- expected vs observed graph delta；
- Skill candidate generation 与验证；
- 图和 Skill 的跨轨迹更新。

但它更像一个**系统蓝图**，没有首先回答清楚：

1. 为什么一定需要图？
2. 为什么一定要让图和 Skill 共同进化？
3. 一次失败为什么不能直接交给现有 Reflection / Skill Optimizer？
4. 论文究竟发现并解决了哪个可被独立评测的新问题？

### 1.2 中间版本

随后方案收窄为：

> 在部分可观测具身执行中，用 Skill 预测的图变化与实际图变化之间的 mismatch，判断应该修正图状态、动作规则还是 Skill。

这一步已经接近正确，但仍有两个问题：

- “实际图变化”在 POMDP 中并不是真实状态变化，只是当前证据支持的 belief update；
- 如果 mismatch 归因和 observed transition 都受 Skill 预测影响，会形成自证循环。

### 1.3 最终收敛版本

最终方案把核心改为：

```text
动作级视觉交互
      ↓
预期状态变化 与 独立证据支持的信念变化不一致
      ↓
先做 update-target credit assignment
      ↓
belief / action model / skill / no persistent update
      ↓
只有可信 Skill defect 才进入字段级修改和候选验证
```

这个版本的关键变化是：

- 研究对象从“graph–skill co-evolution 系统”变成“持久记忆更新的责任分配问题”；
- 图从 novelty 本身变成让预测、证据和 Skill 使用同一语言的 **shared evidence interface**；
- Skill optimizer 从论文主角变成 credit assignment 之后的下游模块；
- `no persistent update` 成为一等决策，而不是没有触发更新时的默认残余类别；
- 实验目标不再只是最终成功率，而是同时验证归因准确性、更新收益和有害更新率。

---

## 2. 最终问题定义

### 2.1 真实现象：一次失败不等于 Skill 错了

在长程、部分可观测、多实例具身任务中，同一个失败表面可能来自完全不同的原因。

以当前 Motivation 图中的任务为例：

> **Instruction:** Deposit all apples onto the TV stand.  
> **Skill:** For tasks containing “all” or “both”, maintain a checklist of remaining target objects.

如果 Agent 没有完成所有苹果的搬运，至少可能存在：

| 失败来源 | 例子 | 正确响应 |
|---|---|---|
| **Episode belief error** | 把 `apple_1` 和 `apple_2` 混为同一实例；把暂时不可见误判为已不存在 | 重观察或修复当前 episode belief |
| **Persistent action-model error** | 不知道 `pick` 需要 near / visible / empty hand；错误理解动作反馈 | 修改跨任务复用的动作前置条件或效果知识 |
| **Reusable skill defect** | “all”任务的 checklist 逻辑错误；遗漏逐实例确认或错误终止 | 修改该 Skill 的具体字段 |
| **Executor lapse** | Skill 已明确要求检查剩余实例，但冻结 VLM 没有照做 | 保留 canonical Skill，短期强调或重试 |
| **Insufficient evidence** | 苹果被遮挡，动作后视野不足，无法确认是否已放置 | 获取新证据，不做长期更新 |
| **Stochastic/no-op event** | 环境偶发失败或动作没有产生稳定可复现的效果 | 记录事件，但暂不修改持久记忆 |

传统做法容易形成：

```text
错误视觉信念
    ↓
任务失败
    ↓
LLM 总结“Skill 应该怎样改”
    ↓
一次看错被固化成长期错误规则
    ↓
后续任务持续负迁移
```

这说明 Skill evolution 的关键瓶颈，不只是候选文字写得是否好，而是：

> **更新信号的责任归属是否正确。**

### 2.2 POMDP 下的最小形式化

将具身执行建模为部分可观测决策过程：

\[
\mathcal M=(\mathcal S,\mathcal A,\mathcal T,\mathcal R,\Omega,\mathcal O).
\]

在时刻 $t$：

- 环境真实状态为 $s_t$，但 Agent 无法直接访问；
- Agent 观察第一视角图像 $x_t$；
- 执行动作 $a_t$；
- 接收环境反馈 $f_t$；
- 维护外部 belief $b_t$，在实现中对应 episode interaction graph $G_t^{epi}$。

当前 Skill $\omega_t$ 与动作模型给出预期 belief transition：

\[
\widehat{\Delta b}_t.
\]

独立的视觉 belief updater 只根据动作前 belief、已执行动作、动作后图像和反馈提取：

\[
\Delta b_t^{ev}
=
U_{epi}(b_t,a_t,x_{t+1},f_t),
\]

其中 $ev$ 表示 **evidence-supported**，不是环境真实 transition。

一个 transition event 定义为：

\[
e_t =
\left(
b_t,\omega_t,a_t,x_{t+1},f_t,
\widehat{\Delta b}_t,\Delta b_t^{ev},E_t
\right),
\]

其中 $E_t$ 是显式证据包。

第一层 credit assignment：

\[
z_t^{mem}
=A_{mem}(e_t)
\in
\{
\text{belief},
\text{action-model},
\text{skill},
\text{no-persistent-update}
\}.
\]

只有当：

\[
z_t^{mem}=\text{skill}
\]

时，系统才执行第二层字段归因：

\[
z_t^{field}
\in
\{
\text{activation},
\text{procedure},
\text{effect},
\text{termination},
\text{constraint}
\}.
\]

### 2.3 这里的 Credit Assignment 不是传统 RL 奖励分配

论文需要主动澄清：

- 不是把一个标量 reward 分配给历史动作；
- 不是声称从观测中恢复了真实因果状态；
- 不是形式上的 structural causal identification；
- 而是将一次 transition mismatch 路由到**应该承担持久更新的记忆组件**。

更准确的表述是：

> **memory-update credit assignment under partial observability**

其目标不是估计不可见的真实 $\Delta s_t$，而是：

> **只在独立证据足够支持时做长期写入，并把写入限制在正确的存储层和字段。**

### 2.4 `no-persistent-update` 必须保留内部原因

第一层输出保持四类，便于定义和评测；但系统内部应保留：

\[
z_t^{none}
\in
\{
\text{executor-lapse},
\text{insufficient-evidence},
\text{stochastic/no-op}
\}.
\]

原因是三者虽然都“不应立即改长期记忆”，后续处理不同：

- executor lapse：重试、增加短期执行强调；
- insufficient evidence：移动视角、主动观察、延迟判断；
- stochastic/no-op：等待重复证据、统计稳定性。

这比把 executor lapse 直接作为第一层第五类更简洁，也比把三种情况完全混为 `none` 更有操作性。

---

## 3. Skill 的最终定义

### 3.1 源码调研后看到的三种 Skill 形态

#### Skill-Pro

Skill-Pro 源码中的单个 Skill 本质上是自然语言 Option：

- `name`
- `initiation`
- `policy`（步骤列表）
- `termination`
- 统计、分数和版本信息

核心语义是：

\[
\omega=\langle I_\omega,\pi_\omega,\beta_\omega\rangle.
\]

它有清晰的 activation / procedure / termination，但没有：

- 显式对象实例绑定；
- graph predicates；
- 每步 expected effect；
- 证据 provenance；
- true / false / unknown 的部分可观测语义。

#### EmbodiSkill

EmbodiSkill 不是维护许多 Option 式独立 Skill，而是维护一个固定域手册，主要包含：

- Planning；
- Search；
- Execution；
- Failure Avoidance；
- Execution Notes。

其论文层面的 Skill 为：

\[
S=(S_{body},S_{app}),
\]

其中 body 保存稳定策略，appendix/notes 用于执行提醒。

这启发 VISTA 区分：

- canonical reusable knowledge；
- 对 executor lapse 的短期执行强调。

#### SkillOpt

SkillOpt 把一个紧凑 Markdown Skill 文档视为冻结模型之外可优化的状态。它强调：

- 有限 patch 操作；
- 小步更新；
- held-out selection gate；
- rejected edit history；
- fast / slow / meta 三种更新时间尺度。

它不要求 Skill 必须是 Option，也没有视觉 transition interface，但它为“如何稳定编辑外部 Skill”提供了很强的工程框架。

### 3.2 VISTA-Skill 的五字段定义

最终推荐：

\[
\omega
=
\left\langle
I_\omega,\pi_\omega,\Gamma_\omega,\beta_\omega,C_\omega
\right\rangle.
\]

| 字段 | 作用 | “all apples”例子 |
|---|---|---|
| $I_\omega$ Activation | Skill 何时适用，需要哪些已知谓词 | 任务包含 `all`；至少存在一个尚未确认完成的 apple instance |
| $\pi_\omega$ Procedure | 执行步骤与恢复流程 | 枚举实例→定位→接近→拾取→移动→放置→更新 checklist |
| $\Gamma_\omega$ Expected effect | 每个关键动作或子步骤预期改变哪些谓词 | `place(apple_i, stand)` 后应支持 `on(apple_i, tv_stand)=true` |
| $\beta_\omega$ Termination | 哪些有证据支持的条件表示 Skill 完成 | 所有已绑定 apple instance 均被验证位于 TV stand |
| $C_\omega$ Constraint | 不变量、禁止条件和安全约束 | 不因当前视野缺失删除实例；手中有物体时不能再次 pick |

其中：

- recovery procedure 放入 $\pi_\omega$；
- 安全规则、不变量和拒绝条件放入 $C_\omega$；
- 对“预期世界变化”的声明放入 $\Gamma_\omega$；
- 每个字段都应能被单独定位和 patch。

### 3.3 为什么删除旧六元组中的 $U_\omega$

旧设计为：

\[
\omega =
\langle
Q_\omega^I,\pi_\omega,U_\omega,\Gamma_\omega,Q_\omega^\beta,C_\omega
\rangle.
\]

其中 $U_\omega$ 让 Skill 根据反馈更新图。

这个设计被淘汰，因为它会产生潜在循环：

```text
Skill 预测动作应该成功
       ↓
Skill 自己定义图应该怎样更新
       ↓
更新后的图“证明”Skill 的预测正确
```

最终设计中：

- Skill 只声明 expected effect；
- 唯一有权写入 episode belief 的是独立 $U_{epi}$；
- $U_{epi}$ 可以知道执行了什么动作和收到了什么反馈，但不能看到 Skill 的 predicted delta；
- expected branch 和 evidence branch 只在 attribution 阶段汇合。

> [!important] 这是 Method 最关键的防泄漏设计
> **预测方不能同时充当自己的证据生成方。**

### 3.4 Skill effect 与 persistent action model 如何避免重复

这是 reviewer 很可能追问的点，需要在方法和实验中明确。

#### Persistent action model

表示跨任务、跨 Skill 复用的原子动作知识，例如：

```text
pick(o)
preconditions: near(agent,o), visible(o), hand_empty
effect: holding(agent,o)
```

#### Skill expected effect

表示某个程序在当前任务语义和步骤中的预期结果，例如：

```text
within “deposit all apples”:
place(apple_i,tv_stand)
→ on(apple_i,tv_stand)
→ completed(apple_i)
→ remaining_count decreases
```

归因规则：

- 同一个原子动作在多个 Skill / 任务中都出现一致冲突，更像 action-model error；
- 只在特定 Skill 的步骤映射、目标绑定、checklist、完成条件中出现，更像 skill defect；
- 当前实例状态与证据不一致，更像 episode belief error。

必须通过 fault injection 和 action-only / skill-only ablation 验证这一区分，而不能只靠文字解释。

---

## 4. 最终 Method 总体框架

### 4.1 四类核心状态

VISTA-Skill 维护：

1. **Episode Belief Graph $G_t^{epi}$**  
   当前 episode 的实例级、带证据和不确定性的视觉信念。

2. **Persistent Action-Effect Memory $\bar G^{act}$**  
   跨任务复用的动作前置条件、效果、适用上下文和证据统计。

3. **Procedural Skill Pool $\mathcal W$**  
   由五字段 Skill 组成的版本化、可回滚程序记忆。

4. **Update Control State**  
   包含短期 Execution Emphasis Buffer、rejected update ledger、候选验证历史和 audit 信息。

其中真正不可缺的核心是前三类记忆与 credit assignment；快慢更新、meta-memory 属于增强层。

### 4.2 Episode Belief Graph

Episode graph 的目标不是构建完整场景数字孪生，而是维护当前决策所需的局部信念。

推荐节点：

- object instance：`apple_1`, `apple_2`, `tv_stand_1`；
- agent / gripper；
- task/subgoal；
- action event；
- evidence/view。

推荐谓词：

- 实例属性：类别、颜色、交互状态；
- 空间关系：near、on、inside、visible；
- 执行状态：holding、opened、powered；
- 任务状态：assigned-to、completed、remaining；
- 证据关系：supported-by、last-seen-at。

状态采用三值语义：

\[
\{\text{true},\text{false},\text{unknown}\}.
\]

必须避免：

- `not visible` 被当作 `not exist`；
- `visible` 被当作 `near` 或 `pickable`；
- 一次旧观测永远覆盖新反馈；
- 多个同类对象被错误合并。

### 4.3 Persistent Action-Effect Memory

每条持久规则至少保存：

- action schema；
- preconditions；
- expected effects；
- applicability context；
- positive / negative evidence IDs；
- occurrence count；
- confidence；
- source tasks / object classes；
- parent version 与更新时间。

单次事件默认只写 episode belief，不直接修改持久动作知识。Persistent action rule 的创建或修改必须满足：

- 跨 episode 或跨实例重复；
- 证据来源彼此独立；
- 不是同一轨迹中重复描述同一个事实；
- 候选在 held-out 或 paired replay 中通过验证。

### 4.4 执行阶段

冻结 VLM executor $\pi_\theta$ 接收：

\[
a_t \sim \pi_\theta
\left(
a \mid I,x_t,G_t^{loc},\omega_t
\right).
\]

其中：

- $G_t^{loc}$ 只序列化当前决策相关的局部谓词，控制上下文成本；
- Skill activation 必须由当前已支持的谓词触发；
- 已知 contradiction 可阻止动作；
- unknown 应触发主动观察或保守规划，而不是被当作 false；
- Skill termination 必须由已验证状态决定，不能只靠语言模型说 “done”。

---

## 5. 核心机制：Evidence-Decoupled Visual Transition

### 5.1 两条信息隔离的分支

#### Expected Transition Branch

允许读取：

- 动作前 belief；
- 当前 action；
- 当前 Skill 的 effect / procedure；
- persistent action model。

输出：

\[
\widehat{\Delta b}_t.
\]

它回答：

> **如果当前 Skill 与动作知识正确，这一步预计改变什么？**

#### Evidence-Supported Transition Branch

允许读取：

- 动作前 belief；
- 已执行 action；
- 动作后图像；
- 环境反馈；
- 新视角或传感证据。

禁止读取：

- $\widehat{\Delta b}_t$；
- Skill 写出的 expected effect；
- attribution 结论；
- 候选 patch。

输出：

\[
\Delta b_t^{ev}.
\]

它回答：

> **当前独立证据真正支持哪些 belief 变化？**

两条分支只在 mismatch attribution 阶段相遇。

### 5.2 显式证据包

每个改变的谓词都应附带：

\[
E_t=
\{
\text{predicate},
\text{pre-value},
\text{post-value},
\text{provenance},
\text{confidence},
\text{view-id},
\text{visibility/coverage}
\}.
\]

例如：

```yaml
predicate: on(apple_2, tv_stand_1)
pre_value: false
post_value: true
provenance:
  - environment_feedback: place_success
  - view_id: frame_031
confidence: 0.91
coverage:
  target_visible: true
  receptacle_visible: true
```

证据包的价值：

- 支持自动/人工审计；
- 避免把 LLM 自己生成的解释当作独立证据；
- 为 attribution calibration 提供输入；
- 让 skill patch 可以引用具体 evidence ID；
- 支持 rejected update ledger 和回滚。

### 5.3 Transition mismatch 不能只做二值 Diff

推荐将 mismatch 表示为谓词级集合：

```text
expected-but-unsupported
supported-but-unexpected
contradicted
unknown / uncovered
identity-conflict
timing-conflict
```

特别是：

- expected-but-unseen 不等于 contradicted；
- 对象未进入当前视野应为 unknown；
- 环境明确返回 action failure 时，可以给相关 effect 负证据；
- 多视角 identity 不确定时，应先处理 belief，而不是改 Skill。

---

## 6. Hierarchical Credit Assignment

### 6.1 第一层：确定更新目标

第一层输出：

| Target | 判定重点 | 默认动作 |
|---|---|---|
| `belief` | 事件只与当前实例、视角、状态追踪冲突 | 修复 episode graph / 主动观察 |
| `action-model` | 原子动作的通用前置条件或效果在多上下文重复错误 | 生成 action-rule patch |
| `skill` | 特定程序的 activation / procedure / effect / termination / constraint 有系统性缺陷 | 进入字段级 Skill patch |
| `no-persistent-update` | executor 偏离、证据不足、随机失败或无法区分 | 不写长期记忆 |

实现上不应把所有判断交给一次自由文本 LLM Reflection。推荐：

1. 先用结构化、可审计规则排除明显情况；
2. 再用 teacher/VLM 对剩余歧义事件做受约束分类；
3. 输出标签、置信度、引用的 evidence IDs 和反事实解释；
4. 低置信度自动降级到 no persistent update。

### 6.2 第二层：只定位 Skill 的具体字段

当第一层确认 `skill` 后，第二层才输出：

- `activation`
- `procedure`
- `effect`
- `termination`
- `constraint`

这样可以避免：

- 把视觉 perception error 和 Skill activation 混在一个平级九分类中；
- 一次修改整个 Skill；
- 未被证据涉及的字段被顺带重写；
- patch 发生不可解释的风格漂移。

### 6.3 “无法区分 action model 与 skill”时的保守原则

如果证据不能区分：

- 通用动作知识错误；
- 特定 Skill 对该动作的使用方式错误；

则不应同时修改两者。推荐：

1. 暂存事件；
2. 收集跨 Skill / 跨任务复现；
3. 比较错误是否跟随 action schema 还是跟随 Skill context；
4. 在证据充分后只更新一个主要 target；
5. joint persistent update 只作为受验证候选，而不是默认行为。

---

## 7. 不同归因结果对应的处理

### 7.1 Belief repair

- 重新绑定 object instance；
- 将无充分证据的 false 改回 unknown；
- 主动获取新视角；
- 根据可靠反馈更新 holding / open / on 等状态；
- 只影响当前 episode，不直接写长期 Skill。

### 7.2 Action-model revision

- 聚合同一原子 action 在不同 Skill / object / episode 的证据；
- 生成 precondition / effect / applicability context 的结构化 patch；
- 通过 schema、evidence support 和 held-out replay；
- 记录版本，允许 rollback。

### 7.3 Skill revision

- 先定位五字段之一；
- 聚合同一 Skill、同一字段、相近上下文的独立事件；
- 生成最小 patch；
- 检查 transition consistency；
- 与 parent Skill 做 paired validation；
- 只有显著有益且无关键子群退化时 promotion。

### 7.4 Executor lapse：短期强调，不立即污染 canonical Skill

结合 EmbodiSkill appendix 的启发，推荐维护：

> **Execution Emphasis Buffer**

每条强调：

- 必须绑定具体 Skill ID 与字段；
- 保存触发上下文和 evidence IDs；
- 只在相似上下文激活；
- 有 TTL / 次数上限 / 衰减；
- 不等价于 canonical Skill update。

例如：

```text
For skill deposit-all-objects,
before declaring completion, explicitly re-check the remaining-instance checklist.
TTL: 3 matching episodes.
```

只有当同一问题重复出现，并且证据表明原 Skill 的表达确实不足时，才考虑将其提升为持久 Skill patch。

这比 EmbodiSkill 当前实现中“execution lapse 直接写入 appendix”更保守，因为 appendix 也可能成为未经验证的长期污染。

### 7.5 Insufficient evidence

- 不做持久更新；
- 发出主动观察请求；
- 保存哪些谓词仍 unknown；
- 后续获得新视角后重新构造 event；
- 不能用“没有看到预期变化”直接生成负证据。

---

## 8. Evidence-Gated Skill Evolution

### 8.1 触发条件

普通 Skill defect 只有在以下条件满足时才进入优化：

- attribution 为 `skill`；
- field label 明确；
- evidence confidence 达到阈值；
- 具有可追踪 provenance；
- 在跨 episode / 实例中重复，或属于高置信严重 contradiction；
- 不是同一原始证据的重复转述。

### 8.2 Patch schema

推荐每个候选补丁显式表示为：

\[
p=
(\text{skill-id},\text{field},\text{operation},
\text{old},\text{new},\text{evidence-ids},\text{scope}).
\]

允许操作：

- append；
- insert-after-exact-target；
- replace-exact-target；
- delete-exact-target。

结构安全要求：

- target 不存在时必须失败，不能静默 fallback 为 append；
- 只允许改被归因的字段；
- `old` 必须与 parent version 精确匹配；
- patch 必须引用独立 evidence IDs；
- patch scope 必须声明是 object-specific、task-family 还是通用规则。

### 8.3 Bounded field-level edit budget

借鉴 SkillOpt 的 textual learning rate，但改为字段预算：

- 每轮最多修改 $K$ 个字段；
- 每个字段最多接受 $M$ 个原子操作；
- 初期可以 $K=1$，证明 credit assignment 后再放宽；
- 随着版本稳定，预算衰减；
- 不允许通过一次自然语言重写绕过 patch budget。

这使“编辑幅度”成为可控变量，也让 attribution 的价值更容易隔离。

### 8.4 候选验证流水线

所有持久更新统一经过：

```text
1. Schema check
2. Exact-target patch check
3. Evidence support check
4. Transition-consistency check
5. Paired same-seed replay
6. Parent vs candidate bootstrap confidence bound
7. Protected subgroup regression check
8. Promotion / rejection / rollback record
```

推荐接受标准：

\[
\operatorname{LCB}_{1-\alpha}
\left[
J_{dev}(\omega')-J_{dev}(\omega)
\right] > 0,
\]

并且：

- 任一受保护任务类别没有超过阈值的显著退化；
- dev episodes 与触发更新的 acquisition episodes 分离；
- final audit/test 从未用于候选选择。

样本量很小时可先采用 SkillOpt 风格的严格改进门：

\[
J_{dev}(\omega') > J_{dev}(\omega),
\]

但正式实验应报告配对置信区间。

### 8.5 Rejected Update Ledger

每个被拒绝的候选保存：

- attribution target / field；
- confidence；
- supporting evidence IDs；
- old/new patch；
- 拒绝原因；
- parent/candidate 分数；
- 哪些任务 improved / regressed / persistent-fail；
- 是否与历史拒绝模式重复。

它的作用不是让模型反复尝试同一修改，而是：

- 防止重复生成已知有害 patch；
- 识别持续出现但尚未充分支持的模式；
- 为 slow consolidation 提供审计材料；
- 支持论文中的 update lineage 可视化。

注意：

> **support count 必须统计独立证据源，不能直接用 minibatch size 代替。**

---

## 9. Fast / Slow / Meta：推荐但分优先级实现

### 9.1 Fast layer（核心）

每个关键 action 后执行：

- episode belief update；
- expected/evidence transition 构建；
- mismatch detection；
- no persistent / temporary emphasis 决策。

### 9.2 Slow layer（P2 增强）

跨多个 episode 聚合：

- recurring action-model conflicts；
- recurring field-level Skill defects；
- improved / regressed / persistent-fail pattern；
- 候选持久 patch。

Slow update 必须与 fast update 一样经过 gate，不能因为“是慢更新”就直接写入 current Skill。

### 9.3 Meta layer（可选）

只优化“如何产生和选择补丁”的策略，例如：

- 哪种证据组合更可靠；
- 哪类 patch 经常被拒绝；
- 什么上下文容易造成 false attribution。

Meta memory 只供 optimizer 使用，不在执行时直接注入 Skill，避免上下文膨胀和未经验证的规则影响 executor。

> [!warning] 论文主线不应依赖 Meta layer
> 若时间不足，Meta skill、复杂 slow optimizer 和完整双向 co-evolution 都应后置。核心贡献必须在最小 VTCA 系统上先成立。

---

## 10. 图在论文中的最终角色

场景图不是贡献本身。动态图、状态感知图、符号世界模型和执行反馈更新都已有前沿工作。

VISTA 中图的准确定位是：

> **Shared evidence and execution interface**

它让以下内容使用同一谓词空间：

1. Skill 是否适用；
2. action precondition 是否满足；
3. Skill / action model 预测什么变化；
4. 动作后证据支持什么变化；
5. 何时验证 Skill 完成；
6. mismatch 应该归因给哪种 memory。

图的价值不是“信息更多”，而是：

- 提供 instance-level binding；
- 显式表达 unknown；
- 让 predicted transition 与 evidence-supported transition 可比较；
- 让 patch 可以引用具体谓词和证据；
- 让 update target attribution 可标注、可测量、可审计。

论文中不要声称：

- first dynamic scene graph；
- first graph-based embodied planner；
- first system updating graph from action feedback；
- first graph–program joint feedback。

---

## 11. 与 Skill-Pro、EmbodiSkill、SkillOpt 的全面对比

### 11.1 Skill-Pro

#### 可借鉴

- activation / procedure / termination 的 Option 式结构；
- semantic gradient；
- multi-trajectory aggregation；
- candidate generation；
- PPO-style gate；
- Skill pool score / version maintenance。

#### 局限

- 主要是文本/语义 trajectory；
- 没有独立的视觉 transition evidence；
- semantic gradient 默认在 Skill 内部找修改方向；
- PPO gate 评价候选对历史 action / return 的相对适配，不等价于物理状态验证；
- 多实例身份和部分可观测 unknown 主要靠语言上下文隐式维持。

#### VISTA 的关系

```text
VISTA attribution frontend
          ↓
筛出可信 Skill defects
          ↓
Skill-Pro-style semantic gradient / candidate generation
          ↓
candidate gate + transition consistency
```

> **Skill-Pro optimizes a skill from experience; VISTA-Skill decides whether the experience should optimize the skill at all.**

### 11.2 EmbodiSkill

#### 可借鉴

- 失败不一定是 Skill defect；
- Skill body 与 execution note 分离；
- success / failure 使用不同 reflection 逻辑；
- execution note 应绑定现有手册规则。

#### 源码审计看到的限制

- 当前开源实现集中在 ALFWorld 文本环境，尚没有完整释放论文中 EmbodiedBench 视觉版本；
- 整个 manual 每步注入；
- reflection 主要查看整条清洗后的文本 trajectory；
- 没有独立 visual transition branch、provenance 或 confidence；
- `FAIL_EXECUTION` 若不能 exact-match manual rule，当前代码会降级成 `FAIL_SKILL`，而不是 abstain；
- 多实例轨迹处理中会移除数字，可能破坏 `apple1/apple2` 身份；
- execution note 的持久化没有与 body candidate 完全等价的 held-out gate。

#### VISTA 的关系

EmbodiSkill 问：

> **Skill body 需要改，还是 executor 需要提醒？**

VISTA 进一步问：

> **在动作级视觉证据上，错误首先属于 belief、action knowledge、skill，还是根本证据不足？**

### 11.3 SkillOpt

#### 可借鉴

- Rollout → reflection → aggregation → rank edits → patch → gate；
- append / insert / replace / delete 的结构化操作；
- bounded edit budget；
- held-out validation strict gate；
- rejected edit buffer；
- fast / slow / meta 分层；
- train / val / test 分离。

#### 源码审计看到的风险

- 默认输入仍是文本 trajectory，没有视觉状态转移证据；
- EmbodiSkill-style appendix 可能在 body candidate 被拒绝后仍被写入；
- slow update 默认可绕过 selection gate；
- insert target 缺失时可能 fallback append；
- support count 有时只是 minibatch 大小，不是真实独立证据数；
- 反复使用很小的 validation set 选择候选，仍有 selection overfitting 风险。

#### VISTA 的关系

SkillOpt 问：

> **如何稳定地优化一份外部 Skill 文档？**

VISTA 问：

> **哪些 action-level visual events 有资格成为这份优化器的训练信号？**

因此 SkillOpt 最适合作为：

- VISTA 的下游 patcher / gate 参考；
- 强 baseline；
- 控制实验中的 common updater。

### 11.4 最强组合 baseline

不能只与 Skill-Pro-VLM 比较。调研后推荐加入：

> **SkillOpt-SAR-VLM**

即：

- SkillOpt bounded optimizer 与 held-out gate；
- EmbodiSkill 的 Skill Defect / Execution Lapse routing；
- 与 VISTA 相同的 VLM、图像和环境反馈；
- 但没有独立 transition evidence 与四类 memory target attribution。

这个 baseline 能回答：

> 即使给现有方法更强的 skill-aware routing 和更稳定的 optimizer，VISTA 的视觉 credit assignment 是否仍然带来增益？

---

## 12. 真正可以主张的创新

### 12.1 核心创新 1：问题层

首次把论文重点收敛到：

> **部分可观测视觉具身执行中，面向持久程序记忆更新的 action-level visual transition credit assignment。**

注意 novelty 不应写成泛化的“first credit assignment”，而要包含限定：

- visual；
- action-level transition；
- partially observable embodied execution；
- heterogeneous external memory targets；
- procedural skill evolution。

### 12.2 核心创新 2：方法层

> **Expected transition 与 evidence-supported transition 的信息隔离。**

这是避免自我验证的关键机制，也使 attribution 有更可信的证据基础。

### 12.3 核心创新 3：更新层

> **先 target-level、再 field-level 的分层归因，以及与证据绑定的最小 patch。**

相比直接从失败轨迹重写 Skill，它明确限制：

- 是否允许持久写入；
- 写到哪种记忆；
- 改 Skill 的哪个字段；
- 哪些证据支持这次修改。

### 12.4 核心创新 4：评测层

> **不只评测 task success，而是直接测量 attribution accuracy、beneficial-update precision 和 harmful-update rate。**

如果配合结构化 fault injection，这会把“可靠 Skill 进化”变成可证伪的研究问题，而不是只比较最终平均分。

### 12.5 不是核心创新的内容

- training-free：重要约束，不是新颖点；
- frozen VLM：公平设定，不是单独贡献；
- scene graph：表示工具，不是单独贡献；
- Skill patch gate：已有 Skill-Pro / SkillOpt；
- skill defect vs executor lapse：已有 EmbodiSkill；
- graph–skill co-evolution：只有实验满足联合增益条件后才能保留；
- fast / slow / meta：借鉴的优化组织方式，不应包装为主要原创。

---

## 13. Motivation 图的最终规划

保留当前例子不变：

> **Deposit all apples onto the TV stand.**

建议整张图分成三个横向子图，用一个例子讲完“现象—问题—方法答案”。

### 13.1 子图 (a)：Same Failure, Different Causes

推荐标题：

> **(a) One Failure, Multiple Responsible Components**

绘制逻辑：

1. 顶部保留 instruction 和当前 textual Skill；
2. 中部保留现有连续第一视角画面，突出多枚 apple instance；
3. 结尾呈现“未完成所有苹果 / 失败”；
4. 从同一失败分出四个可能原因：
   - stale/uncertain visual belief；
   - wrong action knowledge；
   - defective checklist Skill；
   - executor lapse / insufficient evidence；
5. 不要在这一格直接把结论写成“textual skills lack grounding”，因为这会提前认定 Skill 是问题来源。

这一格要传达：

> **同一条失败轨迹无法唯一证明 Skill 有错。**

### 13.2 子图 (b)：Independent Transition Evidence

推荐标题：

> **(b) Predicted vs. Evidence-Supported Transition**

绘制逻辑：

- 上支路：当前 Skill + action model → expected predicate changes；
- 下支路：before belief + executed action + after image/feedback → evidence-supported changes；
- 在下支路明确画“prediction not visible to evidence updater”或用断开的视觉边界表示信息隔离；
- 用 `apple_1/apple_2`、`on(apple_i, TV stand)`、`remaining(apple_i)` 等少量谓词；
- 用 unknown / supported / contradicted 三种状态，而不是复杂完整图；
- 两条支路最后在一个 mismatch box 汇合。

这一格要传达：

> **VISTA 不用 Skill 自己生成“事实”，而是把它的预测与独立视觉证据比较。**

### 13.3 子图 (c)：Attribute Before Evolving

推荐标题：

> **(c) Attribute Before Persistent Update**

绘制逻辑：

1. mismatch 进入四路 router：
   - repair belief；
   - revise action model；
   - patch Skill field；
   - no persistent update；
2. 只有 Skill 路径继续展开五字段；
3. 显示“recurrence + evidence confidence”；
4. 再接一个很小的 candidate verification；
5. 用绿色 promotion、灰色 abstain、红色 rejection；
6. 不在 Motivation 图展开完整 optimizer、ledger、slow update。

这一格要传达：

> **先判断该不该改 Skill，再判断改哪里，最后验证是否真的有益。**

### 13.4 Motivation 图应删除或弱化的元素

- 删除“大量节点构成的复杂 graph”作为视觉中心；
- 删除 “graph–skill co-evolution” 大循环；
- 不把图画成论文唯一 novelty；
- 不在一张 Motivation 图塞 candidate ranking、LCB、版本树；
- 不把一次失败直接连到 Skill update；
- 不把 `not visible` 画成 `not exist`。

---

## 14. Method 图的最终规划

Method 图需要比 Motivation 图详细，但仍以一条主数据流为中心。建议横向四个区域，底部增加审计/慢更新通道。

### 14.1 Region I：Graph-Grounded Execution

推荐标题：

> **I. Grounded Skill Execution**

内容：

- Instruction + current image；
- episode belief graph；
- persistent action-effect memory；
- active five-field Skill；
- local graph retrieval；
- frozen VLM executor；
- executed action。

重点：

- Skill activation、precondition、termination 都在同一 predicate space；
- episode graph 和 persistent action model 必须画成不同容器；
- Skill 不应有 observed-state updater $U_\omega$。

### 14.2 Region II：Evidence-Decoupled Transition Construction

推荐标题：

> **II. Independent Transition Construction**

画成上下两支：

**上支 Expected**

```text
belief + action + active Skill + action model
→ predicted transition
```

**下支 Evidence**

```text
belief + action + next image + feedback
→ evidence packet
→ evidence-supported belief transition
```

视觉上必须强调：

- predicted effect 不进入 evidence updater；
- 两者直到 attribution 前不连接；
- evidence 包含 provenance 和 confidence。

### 14.3 Region III：Hierarchical Credit Assignment

推荐标题：

> **III. Memory-Target and Field Attribution**

第一层四类：

- episode belief；
- persistent action model；
- procedural skill；
- no persistent update。

第二层只从 Skill 分支展开：

- activation；
- procedure；
- effect；
- termination；
- constraint。

No-update 分支可用小字标注：

- lapse；
- insufficient evidence；
- stochastic/no-op。

### 14.4 Region IV：Evidence-Gated Evolution

推荐标题：

> **IV. Constrained Patch and Paired Verification**

流程：

```text
recurring evidence cluster
→ bounded field patch
→ schema/evidence/transition checks
→ paired parent-vs-candidate replay
→ LCB + subgroup regression gate
→ promote / reject / rollback
```

对 `belief` 和 `action model` 也画对应的受限更新箭头，但不要让每个 event 同时写三个 memory。

### 14.5 底部通道

推荐两个小模块：

1. **Execution Emphasis Buffer**  
   接 no-update / executor-lapse，只短期影响执行。

2. **Rejected Update Ledger & Slow Consolidation**  
   保存被拒绝 patch 和跨 episode recurring pattern；所有 slow patch 重新进入同一个 gate。

### 14.6 颜色和图例

建议固定：

- 蓝色：当前 episode belief；
- 紫色：persistent action knowledge；
- 橙色：procedural Skill；
- 绿色：independent evidence / accepted update；
- 灰色：no persistent update / unknown；
- 红色：contradiction / rejected patch。

实线表示执行数据流，虚线表示跨 episode 写回，红色禁止符号表示 prediction leakage 被阻断。

---

## 15. 实验设计：如何证明 VTCA，而不只是证明系统更复杂

### 15.1 研究问题

#### RQ1：任务性能

> Visual transition credit assignment 是否在长程任务上提升 success / progress，并减少 invalid action 和 premature termination？

#### RQ2：归因质量

> Evidence-decoupled VTCA 是否比 Skill-Pro semantic gradients、EmbodiSkill trajectory routing 和 SkillOpt skill-aware reflection 更准确识别真正有益的更新目标？

#### RQ3：更新可靠性

> VISTA 是否提高 beneficial-update precision、降低 harmful-update rate，并减少无效 teacher / candidate evaluation 成本？

#### RQ4：组件作用

> Independent evidence、hierarchical attribution、field-level patch、rejected ledger 和 candidate verification 各自贡献多少？

#### RQ5：联合更新是否真的必要

> Skill-only、action-model-only、independent dual update 与 coupled update 哪个更好？只有 coupled 显著最好时才保留 co-evolution 术语。

### 15.2 Baseline 阵容

#### 基础执行

1. Direct frozen VLM；
2. Static Skill；
3. Raw trajectory / reflection memory。

#### Skill evolution

4. Skill-Pro-VLM；
5. EmbodiSkill-VLM；
6. SkillOpt-VLM；
7. SkillOpt-SAR-VLM；
8. VISTA w/o VTCA：使用相同下游 updater/gate，但让所有失败进入 trajectory reflection；
9. Full VISTA-Skill。

#### 受控实验

为了证明提升来自 attribution frontend，而不是更强 optimizer，需要额外做：

> **固定 candidate generator、edit budget、teacher、selection gate，只改变经验选择与归因方式。**

例如统一使用 SkillOpt-style bounded patcher + paired gate：

| Attribution frontend | Common updater/backend |
|---|---|
| unconditional failure reflection | 相同 |
| Skill-Pro-style semantic gradient | 相同 |
| EmbodiSkill skill/lapse routing | 相同 |
| flat mismatch classifier | 相同 |
| VISTA hierarchical VTCA | 相同 |

原生 pipeline 结果和 common-updater 控制结果应分开报告。

### 15.3 公平性控制

所有主要对比固定：

- frozen executor；
- evolution teacher；
- images 与 environment feedback；
- action space；
- initial Skill / Skill token budget；
- acquisition trajectories；
- candidate count；
- edit budget；
- validation episodes；
- teacher calls / generated tokens；
- final evaluation protocol。

否则 reviewer 可以把增益解释为：

- VISTA 看到了更多图像；
- VISTA 使用了更强 teacher；
- VISTA 生成了更多候选；
- VISTA 多跑了更多环境 episode；
- baseline 的 Skill schema 更弱。

### 15.4 结构化 Fault Injection

这是直接评测问题定义的关键。

#### Belief faults

- 删除一个 object instance；
- 合并 `apple_1/apple_2`；
- 交换 instance identity；
- 将 unknown 错设为 false；
- 注入 stale holding / on / open state。

#### Action-model faults

- 删除 `near` precondition；
- 错写 empty-hand constraint；
- 错写 pick/place effect；
- 错配 failure feedback。

#### Skill faults

- activation 错误；
- checklist/procedure 遗漏；
- expected effect 错误；
- premature termination；
- recovery/constraint 错误。

#### No-persistent faults

- executor 故意跳过正确步骤；
- occlusion / insufficient view；
- stochastic action failure；
- no-op feedback。

每个 fault 有 simulator-assisted ground-truth target 和 field，允许计算：

- target Macro-F1；
- field Macro-F1；
- abstention quality；
- confidence calibration；
- confusion matrix。

### 15.5 核心指标

| 指标 | 回答的问题 |
|---|---|
| Task Success / Progress | 最终执行是否更好 |
| Invalid Action Ratio | action precondition 是否更可靠 |
| Premature Termination Rate | termination 是否得到视觉验证 |
| Target / Field Macro-F1 | 是否找对更新层和字段 |
| Beneficial-Update Precision | 接受的持久修改有多少真正有益 |
| Harmful-Update Rate | 有多少更新导致整体或子群退化 |
| Abstention Precision / Coverage | 证据不足时是否懂得不更新 |
| Graph/Predicate Transition F1 | evidence branch 是否可靠 |
| ECE / Brier Score | 置信度是否校准 |
| Teacher Calls / Tokens | 进化成本 |
| Candidate Eval Episodes | 验证成本 |
| Performance–Cost Pareto | 同预算下是否更有效 |

Beneficial update 需要以与 selection split 分离的 audit split 定义。Audit split 永远不能被 update mechanism 访问。

### 15.6 必要消融

#### Representation

- raw trajectory；
- flat predicates；
- instance-level graph；
- oracle graph；
- noisy graph。

#### Transition interface

- no expected effect；
- evidence branch 可看到 prediction（故意泄漏）；
- evidence-decoupled full design。

#### Attribution

- unconditional reflection；
- flat multiclass；
- hierarchical target→field；
- no abstention。

#### Execution lapse

- no reminder；
- persistent appendix；
- TTL Execution Emphasis Buffer。

#### Patch/update

- full rewrite；
- bounded field patch；
- no rejected ledger；
- no recurrence requirement；
- no slow consolidation；
- slow consolidation bypass gate；
- full gated pipeline。

#### Candidate gate

- no gate；
- Skill-Pro PPO gate；
- SkillOpt strict held-out gate；
- transition consistency + paired bootstrap LCB；
- no subgroup regression protection。

#### Memory evolution

- belief only；
- Skill only；
- action model only；
- independent dual；
- coupled update。

### 15.7 最关键的可证伪假设

> **在相同候选生成器和验证预算下，VISTA 的 attribution confidence 应比 trajectory-level reflection score 更准确地预测一次持久更新是否会在 frozen audit episodes 上产生正收益。**

如果结果只是最终 task success 小幅提升，但：

- target attribution 不更准；
- beneficial-update precision 不更高；
- harmful-update rate 不更低；

那么“Visual Transition Credit Assignment”主张并没有真正成立。

---

## 16. 数据划分与评测纪律

至少分为：

1. **Acquisition / Train**：收集触发更新的轨迹；
2. **Selection / Dev**：parent vs candidate gate；
3. **Audit**：事后定义 beneficial/harmful update，优化器不可见；
4. **Final Test**：最终冻结评估，只在方案锁定后运行。

需要防止：

- 像某些开源流程一样每轮在 final unseen tasks 上选 best epoch；
- 反复查询很小 validation set 导致 optimizer overfit；
- 用产生 fault 的同一 episode 同时做 candidate selection；
- 同一个证据经 trajectory summary、reflection、aggregation 重复计数；
- 把 final test improvement 写回 Skill。

推荐：

- paired same-seed evaluation；
- 多 seed；
- bootstrap confidence interval；
- protected subgroup；
- update budget 预注册；
- 每次 promotion 保存完整 lineage。

---

## 17. 论文写作的最终故事线

### 17.1 开场现象

程序化 Skill 可以帮助冻结小型 VLM 完成长程任务，但错误或过度更新的 Skill 会产生持续负迁移。

### 17.2 现有进展

- Skill-Pro 说明 Skill 可以通过 semantic gradients 和 candidate gate 优化；
- EmbodiSkill 说明失败可能来自 Skill defect 或 executor lapse；
- SkillOpt 说明外部 Skill 文档可以通过小步 patch 与 held-out gate 稳定优化；
- 视觉 Skill / scene graph 工作说明视觉经验和结构化状态是可用的。

### 17.3 尚未解决的问题

这些工作仍没有在部分可观测视觉执行中明确回答：

> **一次 action-level mismatch 应该由哪一种适应性记忆承担？**

整条失败轨迹不足以区分：

- 当前 belief；
- 通用 action knowledge；
- 特定 procedural Skill；
- executor/noise/evidence gap。

### 17.4 方法答案

- 共享 predicate space；
- expected/evidence 两条隔离分支；
- target→field 分层归因；
- no-persistent-update；
- evidence-bound minimal patch；
- paired held-out verification。

### 17.5 实验证据

论文必须同时给出：

- task performance；
- attribution ground truth；
- update utility；
- harmful update；
- calibration；
- cost；
- matched strong baselines。

### 17.6 推荐题目

当前最稳妥：

> **VISTA-Skill: Visual Transition Credit Assignment for Reliable Procedural Skill Evolution in Embodied Agents**

更短：

> **VISTA-Skill: Evidence-Gated Visual Credit Assignment for Embodied Skill Evolution**

只有 joint update 实验显著成立，才考虑：

> **VISTA-Skill: Co-Evolving Visual Interaction Knowledge and Procedural Skills**

不建议把 `Training-Free` 放到标题中心，因为它是约束而不是主要 novelty。

### 17.7 推荐 Contributions

1. 定义 VTCA 问题，并提供 target/field 标签与可靠更新指标；
2. 提出 evidence-decoupled transition interface 和 hierarchical attribution；
3. 提出证据绑定、字段级、受验证的持久更新机制；
4. 在 EmbodiedBench 上与 Skill-Pro、EmbodiSkill、SkillOpt 及其强组合做公平比较。

---

## 18. 被淘汰或降级的方案，以及原因

| 旧内容                                                      | 最终处理                                    | 淘汰/降级原因                                         |
| -------------------------------------------------------- | --------------------------------------- | ----------------------------------------------- |
| 以 Graph–Skill Co-Evolution 为问题定义                         | 降为可选系统性质                                | 描述了答案而非问题；必须靠 joint-vs-independent 实验证明         |
| 图是主要 novelty                                             | 改为 shared evidence interface            | 动态 scene graph 和状态图已有大量先例                       |
| training-free 是主要 novelty                                | 改为实验约束/优势                               | Skill-Pro、SkillOpt 等已有 frozen/non-parametric 方案 |
| 六元 Skill 含 $U_\omega$                                  | 删除 $U_\omega$，保留五字段                   | Skill 不能同时预测和生成自己的观测证据                          |
| expected delta 与 observed graph delta 直接比较               | 改为 evidence-supported belief transition | POMDP 中真实 transition 不可完整观测                     |
| evidence updater 看到 Skill prediction                     | 明确禁止信息泄漏                                | 会产生 self-confirmation，削弱 credit assignment 有效性  |
| 一次失败直接触发 Skill rewrite                                   | 先做 target attribution                   | 会把 belief/action/executor 问题固化进 Skill           |
| 九类平级 mismatch 分类                                         | 改为 target→field 两层                      | 标注困难、存储层混乱、类别边界不稳定                              |
| 把 executor lapse 与所有 none 完全合并                           | 外部四类，内部保留三种 none reason                 | 不同 none 原因需要不同后续动作                              |
| 所有持久 memory 同时更新                                         | 默认单 target，joint 作为受验证候选                | 容易发生重复表达、相互补偿和责任不可识别                            |
| “已有 Skill 方法都没有视觉”                                       | 删除绝对表述                                  | XSkill 等已使用视觉轨迹                                 |
| “已有动态图都是静态提示”                                            | 删除绝对表述                                  | MomaGraph 等已有状态感知动态图                            |
| “此前没有图和程序共同反馈”                                           | 删除绝对表述                                  | SCOPE 等已有 symbolic world/plan feedback          |
| 仅用 Skill-Pro-VLM 作主要 baseline                            | 加入 EmbodiSkill、SkillOpt、SkillOpt-SAR    | 新相关工作已覆盖 selective routing 与稳定优化                |
| EmbodiSkill execution appendix 直接持久化                     | 改为 TTL Execution Emphasis Buffer        | 未验证 reminder 也可能污染长期执行                          |
| SkillOpt slow update 直接写 current                         | 所有 slow update 进入同一 gate                | 慢更新并不天然正确                                       |
| patch target 缺失时 fallback append                         | target 缺失即拒绝                            | 会破坏 Skill 结构，产生静默错误                             |
| minibatch size 作为 support count                          | 统计独立 evidence IDs                       | 多条文本可能来自同一事实，不能算独立支持                            |
| 只报告 Task Success                                         | 加归因、更新效用和伤害指标                           | 无法证明论文真正解决 credit assignment                    |
| 用最终 unseen/test 选 best epoch                             | 严格 train/dev/audit/test                 | 会产生评测泄漏                                         |
| Embodi taxonomy + SkillOpt gate + scene graph prompt 的拼接 | 不作为最终 Method                            | 只是模块组合，缺少独立科学问题和关键机制                            |
| 一开始实现完整 fast/slow/meta/co-evolution                      | 按 P0/P1/P2 递进                           | 工作量大，容易在核心假设未验证前过度工程化                           |

> [!important] 最关键的淘汰逻辑
> 所有被删除的内容有一个共同问题：它们要么把已有模块当 novelty，要么在证据不足时过早写入长期记忆，要么让论文无法隔离“VTCA 本身是否有效”。

---

## 19. 当前 Tex 与最终方案之间的迁移清单

截至 2026-07-30，当前项目状态是：

### 已基本对齐

- Abstract 已引入 Visual Transition Credit Assignment；
- Introduction 已从 co-evolution 改为“failure 不等于 Skill defect”；
- Motivation caption 已形成三步逻辑；
- Problem Setup 已加入 POMDP、expected/evidence transition 和四类 target；
- Contributions 已开始使用 attribution 和 harmful-update 指标；
- Experiment 已有 Skill-Pro-VLM 和 attribution/verification ablation 的基础。

### 仍混有旧版内容

- Method 仍有六元 Skill 和 $U_\omega$；
- 仍使用 `observed transition / observed graph delta`；
- `Mismatch-Driven Graph-Skill Co-Evolution` 仍是旧版平级归因；
- expected/evidence 两分支的信息隔离尚未写清；
- evidence packet、none subreason、字段级 patch schema 尚未加入；
- candidate gate 尚未吸收 SkillOpt 的 bounded patch / rejected ledger 启发；
- Experiments 仍只把 Skill-Pro-VLM 写成 principal baseline；
- RQ2 尚未加入 EmbodiSkill、SkillOpt、SkillOpt-SAR；
- Related Work 尚未系统加入 SkillOpt；
- Motivation 图片仍是旧版“text Skill lacks grounding”构图；
- Method 图片目前为空白占位；
- 标题仍保留旧 co-evolving 版本作为候选。

### 下一轮 Method 修改的最小顺序

1. 重写 Skill 为五元组，删除 $U_\omega$；
2. 重写 transition section 为两个信息隔离分支；
3. 定义 evidence packet 与谓词级 mismatch；
4. 重写 hierarchical target→field attribution；
5. 加 no-persistent subreason 与 Execution Emphasis Buffer；
6. 重写 constrained patch、recurrence 和 paired gate；
7. 将 co-evolution 改为条件性扩展；
8. 同步更新 RQ、baseline、ablation 与 Related Work。

---

## 20. 实现优先级与 Go/No-Go

### P0：先验证核心问题

必须完成：

- Skill-Pro-VLM 公平 baseline；
- EmbodiSkill-VLM / SkillOpt-VLM 最小可比实现；
- episode instance belief；
- 五字段 Skill；
- expected/evidence 独立分支；
- 四类 target + 五类 field；
- fault injection 或小规模人工标注；
- common-updater 控制实验；
- beneficial/harmful update replay 框架。

P0 成功标准：

- attribution 明显优于 trajectory routing；
- beneficial-update precision 提升；
- harmful-update rate 下降；
- task performance 至少不因保守更新明显受损。

### P1：提高更新稳定性

- bounded field patch；
- evidence IDs；
- rejected update ledger；
- paired LCB gate；
- subgroup regression；
- TTL Execution Emphasis Buffer；
- action-model persistent update。

### P2：扩展系统完整性

- slow consolidation；
- optimizer meta-memory；
- full action-graph/Skill bidirectional update；
- 更大任务覆盖；
- lineage 可视化；
- 更复杂主动观察策略。

### No-Go 条件

如果实验发现：

- independent evidence branch 本身很不可靠；
- target attribution 不优于强 trajectory baseline；
- filtered events 并不能更好预测 beneficial update；
- 保守 gate 只减少更新次数但不降低 harmful update；

则不应强行使用 VTCA 主标题。

可降级方向：

- **VISTA-Guard**：视觉证据门控 Skill update；
- **VISTA-Graph**：面向长程执行的 state verification；
- 或将工作收窄为 benchmark / diagnostic study。

如果 coupled update 不显著优于 skill-only / action-only / independent dual：

- 删除 `co-evolution`；
- 保留 persistent action memory 作为辅助组件；
- 不把 joint update 写入贡献。

---

## 21. CVPR 2027 标准评估

### 不足以达到 CVPR 主会标准的版本

如果最终工作只是：

```text
EmbodiSkill 的 skill/lapse 分类
+ SkillOpt 的 patch gate
+ 一个 scene graph prompt
```

那么更像已有模块的具身适配和工程组合，创新性偏弱。

### 有竞争力的版本

若能完成：

1. 独立视觉 transition evidence；
2. heterogeneous memory target ground truth；
3. target→field hierarchical attribution；
4. matched strong baselines；
5. common-updater controlled comparison；
6. beneficial/harmful update 直接证据；
7. 多实例、遮挡、长程 fault injection；
8. 严格 split 和 paired confidence interval；

则论文的核心不再是“我们拼了一个自进化 Agent”，而是：

> **我们定义并验证了一个现有 Skill optimizer 之前缺失的具身视觉 credit-assignment 问题。**

这具有更明确的科学问题、可证伪假设、方法机制和评测协议，达到 CVPR 2027 冲刺标准的可能性显著更高。

---

## 22. 风险与诚实边界

### 风险 1：图抽取错误会污染 attribution

必须有：

- oracle/noisy graph；
- predicate calibration；
- evidence coverage；
- abstention；
- 人工审计样本。

### 风险 2：Action model 与 Skill effect 难以区分

必须有：

- 跨 Skill fault；
- 跨 object/episode recurrence；
- action-only vs skill-only；
- ambiguous case abstention。

### 风险 3：更保守只是在“少更新”

必须在匹配 update count 或预算下比较，并报告：

- update precision；
- coverage；
- final performance；
- missed beneficial updates。

### 风险 4：Teacher 与 evaluator 自证

尽量：

- simulator-assisted labels；
- 与生成 attribution 不同的 evaluator；
- 小规模人工双标；
- inter-annotator agreement；
- frozen audit split。

### 风险 5：验证成本过高

报告 performance–cost Pareto，并比较：

- full paired replay；
- small proxy dev；
- transition-level cheap checks；
- staged gate。

### 风险 6：新论文持续出现导致 novelty 被压缩

持续关注：

- embodied skill self-evolution；
- external skill optimization；
- visual agent memory；
- test-time adaptation；
- episodic/procedural memory routing；
- state-transition attribution。

论文主张应始终围绕精确限定的 VTCA，而不是宽泛的“Skill evolution”。

---

## 23. 最终决策清单

### 必须保留

- Visual Transition Credit Assignment 问题定义；
- POMDP / belief 而非真实 observed state；
- expected/evidence 信息隔离；
- 四类 memory target；
- no-persistent-update；
- Skill 五字段；
- target→field 分层；
- evidence provenance / confidence；
- constrained minimal patch；
- held-out paired candidate verification；
- strong baselines 和 common-updater control；
- attribution + beneficial/harmful update 指标。

### 推荐保留

- persistent action-effect memory；
- Execution Emphasis Buffer；
- rejected update ledger；
- bounded edit budget；
- subgroup regression protection；
- slow consolidation，但必须过 gate。

### 暂不作为核心

- Meta skill；
- 复杂全局场景图；
- 完整 graph–skill co-evolution；
- PPO 数学形式作为 ours 的主要理论；
- 大规模全自动 causal graph；
- 多个 teacher 组成复杂 agent society。

### 只有实验成立才保留

- `co-evolution` 标题；
- joint persistent update；
- “更低成本”主张；
- “更强 long-horizon performance”主张；
- “可靠性显著提升”的定量结论。

---

## 24. 可用于组会的通俗总结

> 以前的方法已经很会“改 Skill”了：Skill-Pro 会从轨迹生成修改方向并验证候选，EmbodiSkill 会区分 Skill 错误和执行器走神，SkillOpt 会用小步补丁和验证集稳定优化 Skill。  
>
> 但在视觉具身任务里，失败不只来自 Skill。Agent 可能没看清、认错了苹果、忘了手里已经有东西、不懂某个动作的前置条件，或者只是没有照着正确 Skill 执行。如果每次失败都去改长期 Skill，就可能因为一次错误观察学出一条永久错误规则。  
>
> VISTA-Skill 的关键做法，是让 Skill 先说清楚“这一步预计让世界发生什么变化”，再从动作后的图像和环境反馈中独立提取“证据真正支持发生了什么变化”。两者不一致时，系统先判断责任属于当前视觉信念、长期动作知识、Skill，还是暂时不应该长期更新。只有证据反复表明确实是 Skill 某个字段有问题，才生成一个最小补丁，并在独立任务上与旧 Skill 配对验证。  
>
> 所以我们的创新不只是多了一个图，而是在 Skill optimizer 之前增加了一个可解释、可评测的视觉信用分配层，目标是让自进化不仅“会改”，还要“知道什么时候不该改”。

---

## 25. 关联笔记

- [[20260730-VISTA-Skill问题定义与Skill-Pro对比核心Insights]]
- [[20260727-问题定义调研与指导意见]]
- [[20260722-VISTA-GraphSkill-CoEvolution-CVPR2027方案备份]]
- [[20260728-Skill-Pro在Mac与MLX上的复现小结]]
- [[20260624-skillv2_vs_direct_insights]]
- [[20260625-优化EmbodiSkill训练流程的idea]]

---

## 26. 参考论文与源码

### 核心 Skill 进化工作

- [Skill-Pro: Learning Reusable Skills from Experience via Non-Parametric PPO for LLM Agents](https://openreview.net/forum?id=9kJQjx2B80)
- [Skill-Pro source code](https://github.com/Miracle1207/Skill-Pro)
- [EmbodiSkill: Embodied Skill Self-Evolution with Skill-Aware Reflection](https://arxiv.org/abs/2605.10332)
- [EmbodiSkill source code](https://github.com/air-embodied-brain/EmbodiSkill)
- [SkillOpt: Executive Strategy for Self-Evolving Agent Skills](https://arxiv.org/abs/2605.23904)
- [SkillOpt source code](https://github.com/microsoft/SkillOpt)
- [SkillOpt documentation](https://microsoft.github.io/SkillOpt/)
- [XSkill: Continual Learning of Multimodal Agent Skills and Experiences](https://openreview.net/forum?id=AjP1yvCyoG)

### 图、符号世界与评测

- [MomaGraph: State-Aware Spatial-Functional Scene Graphs for Mobile Manipulation](https://arxiv.org/abs/2512.16909)
- [SCOPE: Evolving Symbolic World Models and Refining Plans](https://openreview.net/forum?id=PLJ53zWDTD)
- [SkillsBench](https://arxiv.org/abs/2602.12670)
- [SkillEvolBench](https://skillevolbench.github.io/)
- [EmbodiedBench](https://proceedings.mlr.press/v267/yang25f.html)

### 本次源码审计快照

- EmbodiSkill：`760126030eab1d33ec6a6f30988f0f1fb58df3a7`（2026-07-11）
- SkillOpt：`7da46ae693ee0329b80225c0128a37d65db10e9e`（2026-07-30）
