# VISTA-Skill：进化范式、经验规模与自动增强整合记录

创建：2026-09-07；最近复核：2026-09-08。状态：研究假设与实验建议；本次仅更新文档，没有修改方法代码、实验配置或启动新实验。

本文整合本会话的 `20260907_skillopt_comparison.md`、`20260907_skill_training_data_analysis.md`，以及用户提出的“任务数量 × 每任务重复执行 × 多轮进化”假设。前两份文档保留为入口，后续以本文为统一记录。本文保留原有来源、源码核对、数据盘点、优化建议和研究边界，并将早期 E11 尚未运行的状态更新为最新 No-Go 结果。

这里的 Training 指冻结执行器下的外部 Skill 优化，不是模型权重训练。依据是 2026-08-06 中文设计、当前工作区代码（包括未提交的 recovery 变更）、[implementation](implementation.md) 和 [Phase-5 实验日志](experiment_log_phase5.md)。设计建议不代表当前已经实现，更不代表已证明提升。

## 1. 综合判断与用户提出的研究假设

**用户的想法有道理，应作为下一阶段优先检验的假设：有效经验不足既可能表现为不同任务太少，也可能表现为同一任务执行次数太少，因而缺少成功、失败和恢复路径的对照；Skill 更新后再次执行旧任务，还可能产生前一版本无法到达的新轨迹。**

需要修正的是因果措辞：目前已经验证的是局部学习机制，尚未建立稳健多轮进化；“这是由数据不足造成”仍是待验证的解释，不能写成唯一或已确定原因。候选语法过窄、执行器无法完成恢复、视觉证据不足、验证功效以及优化器未利用失败历史，都可能共同限制进化。

区分三个作用：增加任务/布局多样性扩大情境覆盖；增加同任务有效重复丰富条件内的行为与证据；多轮回访让新 Skill 重新生成经验。三者可能互补。重复次数增加本身不保证轨迹多样性，多样性增加也不自动保证对新任务的泛化；最终都要用独立任务收益和有益更新率验证。

VISTA 已实现的相对长处是视觉证据与预测隔离、细粒度归因、结构化补丁和配对回归控制；SkillOpt 的候选搜索、历史利用和迁移实证更成熟。冻结模型、外部 Skill、bounded edits、held-out gate、冻结导出均不能单独作为 VISTA 的独特贡献。核心问题仍是：怎样以有限真实交互获得足以支持长期规则变化的证据，并产生实际有益的更新。

## 2. SkillOpt / EmbodiSkill 范式与 VISTA 优化位置

SkillOpt 依据 [论文 v2](https://arxiv.org/html/2605.23904v2)（2026-05-25；重点 §3、§4.2–4.5、附录 B/C）及 [Microsoft 官方源码固定版本](https://github.com/microsoft/SkillOpt/tree/79124b37e9a6371e13b753f8bcd7adb1e493ade1)（HEAD 2026-09-06，`79124b37`）。当前源码晚于论文，下面区分论文方法、默认配置和可选扩展。

### 2.1 源码逐项比较

| 维度 | SkillOpt 已核对内容 | VISTA 当前内容与判断 |
|---|---|---|
| 更新信号 | 带得分轨迹；成功/失败 minibatch 分析、层级合并、候选排序。[反思源码](https://github.com/microsoft/SkillOpt/blob/79124b37e9a6371e13b753f8bcd7adb1e493ade1/skillopt/gradient/reflect.py)、[合并源码](https://github.com/microsoft/SkillOpt/blob/79124b37e9a6371e13b753f8bcd7adb1e493ade1/skillopt/gradient/aggregate.py) | 动作级 expected/evidence 分离、typed mismatch、归因后跨独立 episode 聚类。局部证据追踪更明确；对跨步骤策略的搜索较窄。 |
| 失败分流 | 当前已有可选 `use_skill_aware_reflection`：区分 Skill 缺陷与执行失误，后者进入 appendix；默认关闭。[源码](https://github.com/microsoft/SkillOpt/blob/79124b37e9a6371e13b753f8bcd7adb1e493ade1/skillopt/optimizer/skill_aware.py) | `belief_refresh / skill_update / abstain`，后者细分证据不足、执行失误、随机等；持久更新要求 provenance、唯一字段等 identifiability 检查。优势在机制化约束，不能简写成“只有 VISTA 区分执行失误”。 |
| 编辑范围 | 文本编辑数量预算；支持 patch 和 rewrite 模式；保护 slow/appendix 区域；`insert_after` 目标缺失可回退 append。[编辑源码](https://github.com/microsoft/SkillOpt/blob/79124b37e9a6371e13b753f8bcd7adb1e493ade1/skillopt/optimizer/skill.py) | 单字段、精确语句目标、证据绑定，文字和 compiled 规则同版本更新；目标缺失拒绝。更利于审计，但表达能力受当前规则语法限制。 |
| 接受规则 | 主训练器 gate 对 hard/soft/mixed 聚合分数作严格大于比较；本身没有 VISTA 式 task-first bootstrap 与 affected/protected 置信界。[gate 源码](https://github.com/microsoft/SkillOpt/blob/79124b37e9a6371e13b753f8bcd7adb1e493ade1/skillopt/evaluation/gate.py) | 静态与转移检查、proxy/finalist 配对评测；受影响任务收益下界与受保护任务非劣性；shadow 不晋升。统计筛选更细，代价是更多 rollout 和更高功效要求，并非总体安全的数学保证。 |
| 历史学习 | epoch 内失败模式和被拒编辑反馈到下一轮；跨 epoch 的 meta memory 只给 optimizer。[trainer](https://github.com/microsoft/SkillOpt/blob/79124b37e9a6371e13b753f8bcd7adb1e493ade1/skillopt/engine/trainer.py#L619)、[meta memory](https://github.com/microsoft/SkillOpt/blob/79124b37e9a6371e13b753f8bcd7adb1e493ade1/skillopt/optimizer/meta_skill.py) | 已有 lineage、processed fingerprint、shadow snapshot；`PatchGenerator.propose(skill, cluster)` 主接口没有历史失败上下文。保存历史与利用历史改进提案是两个不同能力。 |
| 候选预算 | 对合并编辑按影响、一般性、互补性和可执行性排序，并限制 top-L。[排序提示](https://github.com/microsoft/SkillOpt/blob/79124b37e9a6371e13b753f8bcd7adb1e493ade1/skillopt/prompts/ranking.md)、[实现](https://github.com/microsoft/SkillOpt/blob/79124b37e9a6371e13b753f8bcd7adb1e493ade1/skillopt/optimizer/clip.py) | 每轮候选数量已受限；当前 `_cluster_priority` 主要按 constraint/procedure/termination/activation/effect 固定顺序。可进一步按预计效用和成本分配预算。 |

VISTA 对应入口：[EvidenceRequest](../vista_skill/schemas.py)、[attribution](../vista_skill/attribution.py)、[clustering](../vista_skill/clustering.py)、[evolution](../vista_skill/evolution.py)、[模型补丁生成](../vista_skill/models.py)、[temporal](../vista_skill/temporal.py)、[recovery](../vista_skill/recovery.py)。compiled monitor 已能影响动作准入，但不代表所有自然语言 procedure 都可执行，也不代表其任务收益已成立。

### 2.2 论文、源码及可选扩展的差异

1. **慢更新门控存在默认行为差异。** 论文 §3.6 描述 slow-update 候选也经过验证；当前 [默认配置](https://github.com/microsoft/SkillOpt/blob/79124b37e9a6371e13b753f8bcd7adb1e493ade1/configs/_base_/default.yaml) 为 `slow_update_gate_with_selection: false`。在 [trainer 的对应分支](https://github.com/microsoft/SkillOpt/blob/79124b37e9a6371e13b753f8bcd7adb1e493ade1/skillopt/engine/trainer.py#L2002)，慢更新直接进入 `current_skill`，`best_skill` 保持原先验证快照；后续 final validation 可让末态竞争 best。不能由此声称导出 best 完全不验证，也不能把论文“每次更新先验证”视为当前默认训练路径的不变量。该静态差异不能反推论文实验实际配置。
2. **不能把所有错误分流或非回归机制都归为 VISTA 独有。** 除上表 skill-aware 可选分支，独立的 SkillOpt-Sleep 还支持默认关闭的逐任务 `gate_no_regression`。它是论文之后的伴随系统，不能混入论文主方法或误称为 bootstrap 非劣性验证。[Sleep 文档](https://github.com/microsoft/SkillOpt/blob/79124b37e9a6371e13b753f8bcd7adb1e493ade1/docs/sleep/README.md)
3. **监督信息必须匹配。** 当前 ALFWorld adapter 在数据存在时构造人类步骤、PDDL 参数和高层计划参考，经 base adapter 附到结果，再进入反思的 hidden-reference 区域。这是额外训练监督，并不自动意味着测试泄漏。移植到 EB-HAB 对比时必须明确是否禁用/匹配该通道。[ALFWorld adapter](https://github.com/microsoft/SkillOpt/blob/79124b37e9a6371e13b753f8bcd7adb1e493ade1/skillopt/envs/alfworld/adapter.py#L159)、[base adapter](https://github.com/microsoft/SkillOpt/blob/79124b37e9a6371e13b753f8bcd7adb1e493ade1/skillopt/envs/base.py#L77)
4. **同规模自优化也不是 VISTA 独有。** 论文表 5 有 target-matched optimizer 实验。可检验的目标应是冻结 8B、真实视觉部分可观测交互下的性能与进化成本，而不是仅强调 teacher 与 executor 同规模。

### 2.3 当前优势的证据边界

- 已实现的优势是证据隔离、三路归因、独立复现要求、语义/compiled 一致补丁以及更细的配对回归筛选。规则/公共反馈优先的路径还具备节省视觉调用的条件性优势；纯 RGB 或弱反馈下并未证明同样可靠，不能泛化成“视觉理解更强”。
- SkillOpt 论文的多设置任务收益与迁移证据更完整；VISTA 的受控 fault 诊断不等于对 SkillOpt 的胜率实验，两个环境的成功率也不能直接横比。
- VISTA [Phase-5 E10](experiment_log_phase5.md#e10--p56-untouched-finalist-shadow-confirmation--2026-09-03) 在独立 finalist 上发现旧时序候选 success 从 0.8333 降到 0.7000，affected composite delta 为 -0.198237。该候选有实际回归，不能只解释为 gate 太保守。[E11](experiment_log_phase5.md#e11--p57-bounded-target-evidence-recovery-pilot--2026-09-07) 已完成自然 acquisition 和机制筛选；V1 及单独注册的触发修复重检均 No-Go，没有晋升，held-out 阶段未开启。

## 3. 数据来源、当前库存与经验缺口

- [SkillOpt 论文 v2 §4.1](https://arxiv.org/html/2605.23904v2)：ALFWorld 配置为 39 training、140 selection、134 test。其经验生成、minibatch 反思和验证分开；小训练集不是不能优化外部 Skill，但该结果不能直接推断 8B 视觉任务所需规模。
- [EmbodiSkill 论文 v2 §4.1](https://arxiv.org/html/2605.10332v2)：ALFWorld 3,553 training；EB-HAB 和 EB-NAV 各 1,000 training；默认 10 个修订阶段、每轨迹最多一个反思。任务流随机排序，必要时重复打乱。1,000 是任务池规模，不应自动当作每次运行实际消耗的轨迹数。
- EmbodiSkill 的本地官方源码缓存 `/tmp/embodiskill-audit-20260902/EmbodiSkill-main/tasks/run_epochs.py` 中，`allocate_cyclic_task_indices` 实现循环打乱采样；公开 ALFWorld 启动脚本默认 10 epochs、每轮 50 个任务。它不能证明 EB-HAB/NAV 采用相同执行预算。官方 [Issue #2](https://github.com/air-embodied-brain/EmbodiSkill/issues/2) 仍询问两个 1,000 任务集的来源，当前页面未显示作者给出的生成和划分说明；不能把它们当成本地已可用的数据。
- 实际读取 `EmbodiedBench/embodiedbench/envs/eb_habitat/datasets/train_validation.pickle`：100 episodes、100 个不同的对象布局 hash、86 条不同指令、6 个 `instruct_id`，15 个 `scene_id`，且都属于 `v3_sc0_staging_00..14`。15 个实例不等于 15 种完全独立的房屋拓扑；不同布局也不是无限独立的泛化样本。
- 当前 [Phase-5 配置](../configs/vista_phase5_hab.json) 每个 evolution run 为 60 acquisition / 20 selection / 20 audit；selection 又分为 10 proxy / 10 finalist，各任务三次配对重复。三次重复降低执行随机性，不会将十个任务变成三十个独立任务。
- 实际读取 EB-NAV 五个 JSON 的顶层 `tasks`：每个 60，共 300，均为官方测试子集；未发布 `train_validation`。EB-HAB 的六个官方子集也属于最终评估，不能拿来填训练缺口。[项目数据边界](evaluation_integrity.md)
- 官方另有 [EB-Habitat 轨迹库](https://huggingface.co/datasets/EmbodiedBench/EB-Habitat_trajectory_dataset/blob/main/README.md)，包含多模型在 benchmark 子集的轨迹，卡片建议过 base-train / other-subsets-test 的另一种协议。它不是本项目“全官方子集留作终评”协议下可以直接使用的干净新增训练集；轨迹记录增加也不必然增加独立任务。

### 3.1 已发现的缺口与最新实验

1. **可复用规则的覆盖。** 60 次 acquisition 会漏掉低频的前提条件、遮挡/身份混淆和有效恢复方式。举例：若一种可用事件的独立任务发生率为 1%，60 个任务平均只有 0.6 次，出现至少两次的概率约 12.1%。这是示意计算，不是本项目实测发生率。
2. **成功与失败的辨别性对照。** 反复看到同一种失败不代表知道怎样改才有用。SkillOpt 利用成功/失败 minibatch；EmbodiSkill 区分 discovery、optimization、defect 和 lapse；VISTA 还需要能排除部分可观测替代解释的动作证据。
3. **候选验证的独立任务数。** 10-task proxy 再细分 affected/protected，单侧样本更少。三次执行、更多图像、同义指令都不能补足独立任务数。应在开发集上估计 task-level 差值方差、最低有意义收益和非劣性 margin，再确定预算；不存在通用“1,000 一定够”的门槛。
4. **未被开发反复使用的数据。** [Phase-5 E11](experiment_log_phase5.md#e11--p57-bounded-target-evidence-recovery-pilot--2026-09-07) 的历史 ID 审计发现全部 100 个开发坐标已有 artifacts。新 seed、角色轮换或旧场景新措辞不能恢复全局未暴露性。训练重复使用本身合理，问题是不能再声称这些是研究者从未使用的验证数据。

本会话核对到的最新 E11 结果：12 个新目标组合 acquisition 中发现 25 条合格链，来自 8 个 episode；规则通过自然 recurrence，但 V1 机制筛选 parent 1/2 success、candidate 0/2，随后触发修复的四次重检仍 No-Go。原有 36 个任务生成器改变目标、保留旧开发布局，且角色间源布局不重叠；它已证明组合与采样可以运行，未证明自动进化收益。这里的失败证据量足以产生候选，缺口更像有效恢复过程和可用目标证据，而非单纯“再多收相同失败”。

## 4. 新增假设：任务广度、任务内重复与多轮回访

### 4.1 用不同变量描述经验规模

- `N`：一个 evolution run 中 acquisition 的不同基础任务数。任务来源、父布局和场景另记分组，任务 ID 不自动等于独立统计单位。
- `K`：同一 Skill 版本下，每个基础任务的计划执行次数。
- `E`：训练任务流的回访轮数；冻结 executor 权重，但 Skill 可在预算和 gate 允许时演化。
- `G`：候选提案/验证机会数，与 `E` 和实际晋升次数分别记录。

均匀完整执行时，acquisition rollout 预算约为 `N × K × E`；自适应采样时直接报告各任务/Skill 版本的实际执行次数。总成本还包括候选生成、paired gate、审计、任务生成与失败尝试，不能只比较这个乘积。

当前 [CLI acquisition 循环](../vista_skill/integrations/embodiedbench/cli.py) 每个 evolution run 对所选坐标执行一遍，并在每个 episode 后尝试更新；[Phase-5 配置](../configs/vista_phase5_hab.json) 的三次重复主要属于 gate。不同 evolution seeds 各自重新初始化并独立冻结，不等价于一个 Skill 在相同任务流上连续训练三个 epochs。

### 4.2 为什么重复训练可能有用

1. 同一任务既有成功又有失败时，可以区分有效程序与偶发失误，避免把一次失败过度概括成长期约束。
2. 不同搜索顺序、动作选择或实际观测过程可能暴露此前未覆盖的前提和恢复路径；成功轨迹也能为 optimization 提供证据。
3. 在 Skill 通过验证更新后，回访旧任务可以揭示之前到不了的状态，检查新增规则是否造成卡住、绕路或误终止，并为后续进化产生新的 acquisition 经验。
4. 多次执行可以估计规则采用率、恢复成功率和任务内波动，改善候选排序与“不够确定就继续收集”的决策。

上述是机制假设，不是本项目已有正向结果。VISTA E11 的失败链已经充足却没有产生成功恢复，提示应关注重复执行是否新增成功/失败对照和可用证据，而不是只增加相同失败日志。

### 4.3 三种“重复”不能混为一谈

| 执行条件 | 可能获得什么 | 应如何记录 |
|---|---|---|
| 同任务、同初态、同 Skill、同采样设置 | 可能近乎相同；如有运行波动，可估计重复性 | 精确/近似重放；不预设存在有效多样性 |
| 同任务与初态、同 Skill、预登记的不同执行采样 | 不同行为路径和成败对照；是否有效需实测 | 任务内 repeated rollouts，保留 seed 与采样协议 |
| 同任务、晋升后的新 Skill | 新的行为分布与可能到达的新状态 | 跨版本回访；不是同一固定策略的重复样本 |
| 改变对象布局、视角、初始状态或目标 | 条件变化产生的泛化/归因证据 | 增强任务变体，保留父任务/布局关系 |

当前 executor 为 temperature 0；改变 seed 并不保证产生不同动作，硬件非确定性也不应被当成可靠的探索方法。若要研究非零 acquisition temperature 或其他显式探索设置，应另立并匹配各方法的训练协议，记录与当前主协议的差异；最终评测仍按预先固定的设置执行。

每次真正重启 episode 应清空 episode-local belief、历史和临时强调；只保留该实验允许的已晋升 Skill 与训练侧记忆。拿旧轨迹重复做教师反思属于证据重用，不是新增环境交互。保留生成轨迹时的 Skill/schema 版本，不能把旧版本产生的 expected transition 当成新版本的实时预测。

### 4.4 任务内证据价值与跨任务独立性同时成立

“重复轨迹不是新任务”不意味着它没有训练价值。重复可以增加一个任务内的证据强度，但不能单靠重命名 episode ID 就满足跨任务/父布局的泛化支持要求。

当前 [clusterer](../vista_skill/clustering.py) 主要依据不同 `episode_id` 和唯一 evidence IDs 统计独立支持。若以后引入训练重复，应显式区分 `base_task_id`、`parent_layout_id`、`rollout_id`、`skill_version` 和 `sampling_seed`：保留每次真实尝试的独立 provenance，同时分别报告任务内复现与跨任务复现。该分组扩展是实施前要求，本次没有改动代码。

验证时先在任务内汇总 paired repeats，再按基础任务/父布局做组级不确定性估计；有场景层级时还需相应的分组/外推分析。只增加 K 不能解决情境覆盖和历史验证暴露，随机重划分也不能恢复全局未见数据。

直观上，总评估波动包含任务间差异与任务内随机差异；增加 K 主要降低后者，增加 N 才能同时改善情境覆盖及任务间均值估计。训练中的最佳 N/K 分配未必等于验证中的分配，不能由 gate 的 paired-3 结果直接推出 acquisition 应重复三次。

## 5. 从经验到有效提案的优化优先级

**P1：把拒绝记录变成提案学习信号。** 在现有 lineage 上增加紧凑的 optimizer-only 历史：规则/父版本、适用条件、尝试的修复、harmful / underpowered / not-adopted 等可审计原因。仅给候选生成与排序，不给 evidence branch。首次可用确定性摘要，避免直接增加长 meta prompt。已确认有害与证据不足必须分开；独立 audit/test 的反馈不得回流提案，selection 也应预先限定反馈粒度。现有 fingerprint 继续防止无新信息的重复评测。

**P2：用短时序成功/失败对照，学习完整恢复过程。** VISTA 已会从成功转移 discovery 动作效果，不能称其“只学失败”。真正缺口是：在相近触发状态下，比较失败链与成功恢复链，识别观察、换位置、重新接近、目标证据确认等有助于完成任务的过程。SkillOpt 的成功/失败分批处理可供参考，但 VISTA 应保留动作证据和跨 episode recurrence。只新增“禁止重试”容易让代理停滞；应生成有触发、恢复动作、解除条件和次数预算的 procedure。P5.7 是此方向的窄语法原型，尚未证明通用过程发现。

**P3：从固定字段顺序升级为效用/成本排序。** 保持主对照的单字段补丁和候选上限，先按独立支持、预计受影响任务覆盖、执行采用证据、风险与 paired rollout 成本排序；必要时才让教师比较候选。不要直接照搬较大 batch、编辑数或增加多个模型调用。Phase-5 E5 中 proxy 占 executor tokens 约 82%，因此减少无效验证比继续压低已为零的 patch-teacher 调用更关键。

**P4：验证局部修复稳定后，再做慢速组合与压缩。** 借鉴相同训练任务上前后 Skill 的纵向对照，把稳定收益、退化、持续失败、稳定成功分开，整理紧凑的规则组合。组合效用不等于各规则效用之和，需要独立验证；禁止绕过 gate 把 slow block 放入 active Skill。VISTA Phase3C 的固定 Meta-Skill 已 No-Go，而 SkillOpt 的 meta memory 是从优化历史生成，两者不等价：前者失败不能证明后者无用，后者在别的设置有效也不能保证 8B 视觉任务受益。既有 Meta 实验的详细结果见 [Phase-3 日志](experiment_log_phase3.md)的 Phase3C 部分。

## 6. 自动增强如何服务上述假设

| 层次 | 操作 | 研究定位 |
|---|---|---|
| 目标与语言组合 | 换目标对象/容器、子目标组合、保持语义的指令改写 | 成本低；基础设施和覆盖增强，对物理/视觉多样性的增益有限 |
| 可执行环境变化 | 在允许的资产/场景中重采样对象位置、初始位置/朝向、可见性和已支持的容器状态 | 产生真实新交互，优先建设；需要物理合法性、动作可解性、目标非初始满足和语言一致性检查 |
| VTCA 驱动的主动生成 | 依据竞争归因和欠覆盖条件选择成组场景变化，采集支持与反驳两类证据 | 可能成为方法贡献，必须超过等预算随机生成、失败优先采样等控制 |

已有基础：EB-HAB 上游 `dataset/generator.py`、`create_episodes.py`、`dataset_validator.py` 和配置包含场景/对象采样、指令语义和搜索验证；当前配置甚至同时列了 train/val/test scene sets，不能直接把默认全量当训练资产使用。上游脚本存在旧路径/依赖，需要适配和 smoke，未在本次运行。通过 `scripts/` 或 adapter 复用，输出写 `running/`，不编辑 stock benchmark。NAV 可参考现有 `Gen_data/base_task_gen.py` 的场景、目标和起点生成，但也须重新建立非官方开发任务及合法性验证。

### 6.1 面向信用分配的实验生成

以一次失败 pick 为例，待区分的解释可能是目标不可见、距离不合适、belief 过期、有效规则未执行、规则本身不完整。生成器不能先将某种解释写成答案，而应构造能使这些解释产生不同可观察结果的任务组。

先在合法父布局中固定目标与任务，分别控制初始视角/遮挡和真实交互前提；在不同父布局上重复。随后由同一个冻结 executor 真实执行，记录公开反馈和 RGB，以及是否遵循当前规则。对恢复学习，还需采到成功解除失败条件的过程。任务对设计只提高辨别力，并不会自动证明唯一因果解释。

主动采样可以用“竞争解释的预测分歧、欠覆盖转移、任务多样性、预计执行成本”作为最初的可计算代理；不必新增一个需要训练的大模型。不能只追求最大失败率，因为不可解任务和没有可观察区分信息的任务会浪费预算。

关键隔离要求：

- 生成器允许依据当前 Skill/归因决定下一次训练任务，但 evidence extractor 仍不读取该假设或预期结论。
- Simulator 内部状态只用于生成与有效性/诊断检查；不把遮挡真值、oracle 计划、隐藏 predicate 等注入 executor 或 VTCA evidence。若未来要使用此类监督，必须显式另立信息条件和对照。
- 所有伪轨迹、语言模型声称的动作结果都不能算真实转移证据。公开反馈充足与弱反馈/RGB 轨道分别报告。
- 同一父布局的多个干预、措辞和 rollout seeds 是一个相关组；recurrence 与统计分析需要父布局/任务组身份，不能把生成的近重复任务当独立支持。
- acquisition 可自适应，selection/audit 的任务分布、生成 seeds 和分组应预先锁定；生成器不能针对它们的结果调任务。最终 Skill 冻结后评估完整官方子集。
- 新生成开发集需登记为明确的新数据协议，记录资产、母任务、变换、可解性、去重和划分来源，不默称仍为原生 60/20/20 实验。

### 6.2 新颖性边界

普通自动课程已有 [Voyager](https://voyager.minedojo.org/)，程序化关卡的学习潜力采样已有 [PLR](https://proceedings.mlr.press/v139/jiang21b.html)，自适应可解环境设计已有 [PAIRED](https://arxiv.org/abs/2012.02096)。这些足以否定“首次自动增强/首次自动课程”的宽泛表述；本次不是穷尽性新颖性审查。

更聚焦的候选贡献是：**通过信用分配驱动的可执行实验生成，减少错误写入并提高固定交互预算下的可靠 Skill 进化。** 它应服务 VTCA 主线，而不是另建不相干 benchmark。

## 7. 统一实验顺序与判据

### 7.1 先检查重复是否产生有效新信息

在 acquisition 内固定 S0 与任务初态，用预登记的重复次数（例如 K=1/3/5）检查：去重后的动作/谓词转移序列、成功/失败共存任务比例、新恢复链和新可用证据覆盖。明确表面措辞变化不算行为多样性。不丢弃失败或只保留最好轨迹；记录所有尝试的成本。

这一步比较的是同版本内重复，暂不让 Skill 更新混入“随机采样是否有效”的判断。如果大部分重复完全相同，先解决探索与可观测性，而非盲目增加 K。

### 7.2 将数量、重复、回访和优化机会分别控制

- **任务广度**：固定 K、回访规则和优化后端，比较 N；后续可沿用前面提出的 100/300/600/1,000 个生成 acquisition 任务档位，validation/audit 另留。它们不是保证充分的数据门槛。
- **任务内重复**：固定同一 N 与任务集合，比较 K=1/3/5，记录总预算增加及单位成本收益。
- **相同 acquisition 预算的分配**：例如在同一合法训练池内比较 60 个不同任务各一次与 20 个任务各三次，均为 60 rollouts；重复多个预登记的分层任务抽样，避免单个 20-task 子集恰好更容易。此对照只用于新诊断，不改变当前正式配置。
- **多轮回访**：在相同 N、总 rollout、提案次数 G 及验证预算下，比较“先在一个 Skill 快照下收集整批经验再更新”与“分轮收集，允许已通过 gate 的更新指导后续回访”。记录每个版本的覆盖；若没有任何候选晋升，不能将差异解释为新 Skill 带来的进化螺旋。

固定样本量比较通常会改变计算成本，因此还要给出等总成本结果。明确候选提案时点和证据去重规则，避免 K 增加同时偷偷增加 G。Gate 必须保持相同标准；不能为了观察多轮更新而强行接受失败候选。

### 7.3 再研究自动生成与自适应分配

比较无增强、等预算随机生成、失败优先采样、归因驱动生成；为用户的重复假设增加“固定任务重复”控制。底层允许的生成操作一致，成本包括任务构造、无效/不可解样本过滤、所有失败执行和后续验证。

生成器可先在 acquisition 上依据未解决归因、恢复覆盖和预计成本分配“继续重复当前任务”或“生成新任务”的预算。只有在固定 N/K 实验建立有效信号后，再检验这一自适应分配是否胜过固定比例；不能把自适应调度天然视为创新或收益。

### 7.4 结果必须回答两个问题

1. 是否获得更多有效经验：支持/反驳证据、成功恢复、独立父任务上的规则复现、跨版本的新状态覆盖？
2. 是否产生更好的 Skill：独立任务成功率、经审计的有益更新比例、子群回归、学习曲线和完整成本是否改善？

更新率、晋升次数、轨迹多样性、训练成功率或 pass@K 均不能单独证明泛化。保留固定的 protected/selection/audit 分布；不能因训练样本更多就增删验证难例。历史 official-test 及独立 audit 的结果不得回流训练采样或提案。

## 8. 对照范式与贡献归属

演化算法比较时，EmbodiSkill*、SkillOpt* 和 VISTA 先共享同一生成训练池与采样预算；保持既有 EmbodiSkill 控制。尤其比较 `SkillOpt*`、`SkillOpt* + VISTA Common Gate`、`VISTA`：星号明确表示 EB-HAB 适配，Common-Gate 两组还需匹配表示、执行约束与更新后端，才更容易归因到 VTCA；原生 SkillOpt 编辑/gate 则作为端到端方法对照。

自适应闭环的比较再允许各方法按自身状态选择新任务或重复任务，同时匹配底层生成权限、executor、teacher、S0、信息条件、提案与总成本。更多有效数据与更有效利用相同数据是两个待分别测量的效应。

若仅增加 N 或 K 就让所有方法获得相近收益，这是有价值的规模/预算结论和基础设施贡献；不能自动算 VISTA 专属算法创新。若信用分配驱动的生成/重复分配在等预算下进一步提高独立任务表现及有益更新率，且消融支持归因机制的作用，才有证据将它列为方法贡献。

近期优先级：先确认同任务重复能否新增可用经验，再在固定预算下比较任务广度与重复深度，同时补足成功恢复对照；随后研究多轮回访和自动生成。数据不足保留为主要候选解释，与执行器、证据和规则表达的瓶颈共同检验。

## 9. 2026-09-08 源码复核：SkillOpt 哪些设计能改善当前瓶颈

本节针对最新 E11 及 UNKNOWN 重检重新排列实施优先级。复核官方仓库的 commit 列表与本地源码缓存，使用的 SkillOpt 版本仍为 `79124b37e9a6371e13b753f8bcd7adb1e493ade1`。以下将源码已有能力、VISTA 的当前缺口和建议改动分别说明；建议均未实施或验证。

### 9.1 当前最直接的缺口是修复方案，而不只是失败样本量

E11 已有 25 条合格失败链、8 个支持 episode，足以通过当前候选生成要求。V1 和触发修复重检均是 parent 1/2、candidate 0/2；两次候选运行都累计三次 guard block 后终止。重检恢复了 UNKNOWN 后的视觉查询，却仍未取得目标的可靠 `near=TRUE`。因此，当前证据支持的具体判断是：**系统识别出无效重试并执行了限制，但还未找到能完成恢复的程序。** 这两任务的机制诊断不构成总体性能估计，也不能证明数据量已经充分。[实验记录](experiment_log_phase5.md#e11--p57-bounded-target-evidence-recovery-pilot--2026-09-07)

源码中的关键断点：

- [recovery.py 的 `recovery_chains`](../vista_skill/recovery.py) 收集“失败 pick → 成功 nav → 同目标再次失败 pick”；若出现成功 pick 或可靠的 `near=TRUE`，只终结待匹配链，没有把这段过程保存成成功恢复范例。
- 同文件的 `propose_recovery` 将符合要求的链映射到一条固定的英文 procedure 和固定 `TemporalSkillRule`。自然数据决定是否准入及引用哪些证据，但没有从不同恢复路径中选出策略。更多相同失败链会增强复现支持，不能让这个生成函数自动学会更有效的搜索方法。
- 这些链能反驳“导航成功足以保证抓取恢复”，却不能单独证明“任何重试前必须取得显式 near 真值”就是最佳处理，更不能证明执行器能在预算内获得该证据。

这说明需要区分**缺陷定位**与**修复方案搜索**：VTCA 确定哪个字段有可复用问题，并不意味着观察到的失败已经唯一决定正确程序。单字段可写与字段内存在多个可检验方案并不冲突。

### 9.2 优先借鉴一：成功/失败轨迹反思，补上恢复过程学习

**SkillOpt 已有：** [`run_minibatch_reflect`](https://github.com/microsoft/SkillOpt/blob/79124b37e9a6371e13b753f8bcd7adb1e493ade1/skillopt/gradient/reflect.py#L485) 分别组织成功、失败轨迹，分析器读取当前 Skill 与轨迹；[ALFWorld success prompt](https://github.com/microsoft/SkillOpt/blob/79124b37e9a6371e13b753f8bcd7adb1e493ade1/skillopt/envs/alfworld/prompts/analyst_success.md) 要求从多个成功案例中提炼已有 Skill 未覆盖的通用行为，包括有效探索。它不是现成的状态匹配成功/失败因果对照算法。

**VISTA 的适配：** 保留短动作窗口中的成功恢复、失败恢复与未解除状态，按目标类型、触发条件和可观测状态组织对照；正例也应覆盖最终整任务失败但局部恢复成功的情况。VISTA 已有成功单步转移 discovery，新增的是跨步骤过程学习，而非首次利用成功数据。用对照生成有明确触发、可选恢复动作、解除证据、搜索次数与退出条件的 procedure；只允许使用环境真实提供的动作。

例如需要学到的是“哪一次导航/搜索改变了目标观测，并产生了足以重试的证据”，而不是只增加“禁止重复”的句子。若真实轨迹没有成功恢复或无法可靠观察解除条件，应记录这个缺口，优先采集相应 acquisition 经验，不让教师补写未经执行的成功过程。

还有一个低成本改进：**完整 Skill 可读、归因字段可写。** [当前 `JsonBoundedPatchGenerator`](../vista_skill/models.py) 的普通模型路径主要传入目标字段语句、局部 mismatch 和该字段 compiled rules，缺少完整程序上下文、恢复窗口和编辑历史。可以给提案器只读的全 Skill/动作契约摘要与相关短窗口，仍保持唯一字段、证据引用和 compiled 一致性检查。这不会要求 evidence branch 读取 Skill，也不必开放整份 Skill 重写。

### 9.3 优先借鉴二：让拒绝记录真正改变下一次提案

**SkillOpt 已有：** [`_format_step_buffer`](https://github.com/microsoft/SkillOpt/blob/79124b37e9a6371e13b753f8bcd7adb1e493ade1/skillopt/engine/trainer.py#L619) 将此前失败模式、被拒编辑及编辑前后分数组织成下一步反思上下文；[训练器](https://github.com/microsoft/SkillOpt/blob/79124b37e9a6371e13b753f8bcd7adb1e493ade1/skillopt/engine/trainer.py#L1645) 持续写入这个 epoch 内的 buffer。这里是经验反馈，不是只防止重复运行同一个文件。

**VISTA 的适配：** [EvolutionCoordinator](../vista_skill/evolution.py) 会写 lineage，也有 `_processed` 去重，但仍调用 `generator.propose(active, cluster)`，未把先前决策反馈给生成器。增加紧凑的提案上下文，区分“已观察到有害”“效果不确定”“未被执行”“证据无法供给”“静态不合法”。先做确定性摘要，保留规则、父版本与证据来源；不必先增加一个长篇 meta prompt。

当前案例可形成的有用教训是：“此恢复模板在开发诊断中阻止重试，同时因解除证据不可得而终止；下一方案需要解决恢复路径或证据供给。”它不应被压缩成“pick 约束都没有用”，也不能把样本不足等同于有害。普通程序改动与运行时 guard 的效应尚未分开，应保留这一不确定性。

反馈只进入 optimizer。可用 acquisition/开发诊断以及预先允许的 selection 汇总；独立 audit、官方 test 的结果不能输入提案，不能倒推已经结束的实验获得新确认效力。相同语义候选没有新假设或新信息时仍应被去重。

### 9.4 优先借鉴三：积累方案、去重排序，再支付验证成本

**SkillOpt 已有：** [aggregate.py](https://github.com/microsoft/SkillOpt/blob/79124b37e9a6371e13b753f8bcd7adb1e493ade1/skillopt/gradient/aggregate.py#L156) 分别合并成功/失败提案，再合并两组；[`rank_and_select`](https://github.com/microsoft/SkillOpt/blob/79124b37e9a6371e13b753f8bcd7adb1e493ade1/skillopt/optimizer/clip.py#L27) 在编辑池超过预算时调用排序器，否则原样返回。[排序规则](https://github.com/microsoft/SkillOpt/blob/79124b37e9a6371e13b753f8bcd7adb1e493ade1/skillopt/prompts/ranking.md) 依次考虑系统性影响、互补性、一般性和可执行性。它筛选的是编辑池，并不意味着逐个候选都进行昂贵的 best-of-K 环境评测。

**VISTA 的适配：** 用一个受限的提案调用给同一归因字段产生少量策略不同的方案，先合并等价方案、排除没有恢复动作或不可实现解除条件的方案，再选一个送入现有 gate。可先用独立任务支持、相关条件覆盖、恢复证据、重复性和预计验证成本做确定性排序。来源窗口与实际试用可以支持排序，但不能把教师自评的收益当成已测性能。

主流程的 `_cluster_priority` 目前主要按字段固定排序，E11 pilot 更是只有固定模板一个方案；两条路径都需要明确接入新的提案过程。现有静态检查、机制筛选、paired evaluator 缓存与 proposal budget 已经存在，新增能力应集中在候选内容和验证前选择，避免重复建设。E5 的 proxy 曾占 executor tokens 约 82%，E11 的 patch-teacher 调用已经为零，因此评价标准应是**每单位总成本获得的有益更新**，而不是单独追求零提案调用。

### 9.5 后续借鉴：重复采集与跨版本记忆，服务已可工作的修复过程

| 设计 | SkillOpt 的源码事实 | VISTA 应如何使用 |
|---|---|---|
| 多 epoch 与批内积累 | [trainer](https://github.com/microsoft/SkillOpt/blob/79124b37e9a6371e13b753f8bcd7adb1e493ade1/skillopt/engine/trainer.py#L1132) 每轮重新组织任务并以 current Skill rollout；[split dataloader](https://github.com/microsoft/SkillOpt/blob/79124b37e9a6371e13b753f8bcd7adb1e493ade1/skillopt/datasets/base.py#L431) 按 epoch 打乱训练项，微批次积累后再合并提案。 | 现有主 CLI 每个 run 顺序遍历 acquisition 一次、每 episode 尝试更新。可增加计划的同任务重复、分批提案和晋升后回访，优先填充成功恢复对照；记录 N/K/E/G，保持等总预算，避免把不同 rollout ID 计为不同基础任务。默认四 epoch 是配置，不是视觉任务的最佳值证据。 |
| optimizer-only meta memory | [meta_skill.py](https://github.com/microsoft/SkillOpt/blob/79124b37e9a6371e13b753f8bcd7adb1e493ade1/skillopt/optimizer/meta_skill.py#L33) 用前后 Skill、相同任务的纵向比较和旧记忆生成新的优化指导，不直接写入 executor Skill。 | 先积累真实的有益/有害修改，再压缩“哪些改法在何种条件下有效”。与 Phase3C 固定通用 Meta-Skill 不同；建议放在短期拒绝 buffer 之后，而非当前第一优先级。 |
| 面向 executor 的 slow update | [slow_update.py](https://github.com/microsoft/SkillOpt/blob/79124b37e9a6371e13b753f8bcd7adb1e493ade1/skillopt/optimizer/slow_update.py#L342) 根据跨 epoch 轨迹比较生成较稳定指导。 | 等局部恢复程序取得收益后再研究组合、压缩和交互；所有改变 active Skill 的内容仍过 gate。当前 SkillOpt 默认 `slow_update_gate_with_selection: false`，不能照搬其直接注入 current Skill 的分支。 |

这些机制支持“重复与回访能提高有限数据的利用率”的合理性，但没有证明 EB-HAB 的 100 个任务足够或不足。若反复执行产生同样失败且生成器仍输出固定模板，增加 K/E 不能补上缺失的恢复算法。自动增强应优先生成能检验恢复路径与证据供给的真实任务/轨迹，并用等预算随机增强和固定重复作对照；其可能贡献仍是第 6–8 节描述的信用分配驱动经验生成。

### 9.6 建议的最小下一步与保留项

1. **先测证据是否可获得。** 在新登记的开发任务面板上，以相同模型/观测预算比较现有泛化导航与有界目标搜索，记录视图是否变化、目标是否出现、`near` 是否可判定、是否能实际恢复。不把可见性等同于抓取距离，也不从教师想象中获得真值。
2. **随后测试提案后端。** 以相同 acquisition 窗口比较固定模板、成功/失败窗口提案、窗口提案加拒绝历史；保持 VTCA、Skill 表示、执行器、教师、验证标准及总预算可比。优化器改进在机制消融中也应提供给匹配的受控基线，以区分后端增强与 VTCA 本身的贡献。
3. **分清文本与 guard。** 当前实验二者同时变化。新协议中分别测试 procedure 文本和 compiled guard 的效应及组合，检查真实恢复率、解除证据、guard 耗尽和任务成功；机制指标改善后才进入预登记的独立配对验证。现有两任务用于已完成诊断，不能反复调到通过后声称泛化。

优先保留 VISTA 的证据隔离、三路归因、证据绑定、精确单字段修改和配对回归控制。当前候选的局部任务失败与 E10 的独立回归说明拒绝有实际作用；放宽 gate 不能替代有益候选。SkillOpt 的参考轨迹通道、模型规模与训练预算也需按既有协议匹配。上述借鉴主要增强“怎么提出有效修复”，并不要求削弱“哪些证据允许长期写入”。
