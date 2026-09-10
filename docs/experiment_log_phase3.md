# VISTA-Skill Phase 3 Experiment Log

> 2026-09-08 目录整理：本阶段实验实体已归档至 `running/Phase3/`，路径引用已改写且兼容软链接已删除，原版已备份。详见[实验输出目录说明](experiment_output_layout.md)。

本文件记录 2026-08-24 开始的 Phase 3。Phase 1/2 的 artifact 和结论保持只读；
任何算法、阈值、样本或运行策略调整必须在新结果用于下一轮决策前记录在本文件中。

## 0. Phase 3 总体推进路线（冻结于 2026-08-24）

Phase 2 已经证明真实 Habitat/Qwen/paired-audit 链路可以运行，也观察到可独立验证的
beneficial repair；但当前系统仍是 0 promotion。主要瓶颈依次是 Evidence 的独立可靠性、
混合 mismatch 的归因召回、fault-relevant task 密度和 gate 统计功效。因此禁止直接扩大到
ACQ=60 x 3 seeds、官方 300 tasks、32B 或第二环境，按以下依赖顺序推进。

### 1. Evidence 硬门槛

- 自动构造约 300 个 state-labeled primitive transitions；不使用人工标签。
- 比较 feedback-only、images-only、images+feedback、可靠性守门和 oracle。
- 报告 predicate/contradiction accuracy、coverage、ECE/Brier、selective risk、错误
  `skill_update` 传导和成本。
- 若综合 Evidence 不可靠，停止扩大 evolution；若综合 Evidence 可靠但视觉没有独立增量，
  论文应弱化“visual”主张，把视觉定位为补充信息源。

### 2. VTCA v2

- 将 Phase 2 仅在审计脚本中验证的 provenance-aware mismatch partition/dominance 接入核心
  `CreditAssigner`。
- Phase 2 的 multihold/effect 两类数据只作为开发集；主结果必须来自 held-out fault variants。
- Go 条件：相对 strongest trajectory baseline 的 target/field Macro-F1 至少 +0.10，
  task/campaign bootstrap 95% CI 不跨 0，clean false `skill_update` rate < 5%。

### 3. SkillFaultBench v0

- 建立 procedure、effect、constraint、termination 四类 repairable faults，每类两个 severity。
- clean、execution lapse、occlusion、identity ambiguity、stochastic/no-op、goal coverage、
  missing rule 作为 no-write/out-of-scope controls。
- affected/protected group 在 outcome 前由 fault semantics 定义，acquisition/selection/audit 在
  scene/object/task 上隔离。
- Fault admission：faulty-vs-correct affected effect 95% LCB >= 0.10；protected LCB > -0.05。

### 4. Gate pilot

- 使用 semantic affected/protected gate，而不是只看 global mean 或 opaque subgroup hash。
- 主要条件：affected LCB > 0；protected LCB > -0.05；global utility 为 secondary outcome。
- 先在两个 held-out fault variants、seed 0 上比较 corrupted frozen、EmbodiSkill*+Common Gate、
  VISTA w/o VTCA、Full VTCA v2 和 Oracle VISTA。
- 只有观察到 online accepted + independent-audit beneficial + protected non-inferior 的更新，
  才扩大到三个 evolution seeds。

### 5. Release-grade repair 主实验

- 三个 evolution seeds，严格隔离 acquisition/selection/audit roles。
- 同时报告 equal candidate/evaluation budget 与 equal teacher-token budget。
- 主指标：affected recovery、protected non-inferiority、beneficial-update precision、harmful-update
  rate、missed-beneficial rate、update coverage 和 cost per accepted beneficial update。
- 在 promoted update 数不足约 30 时，不以“0 harmful”宣称安全，只报告置信区间和样本限制。

### 6. Stock EB-HAB 外部有效性与后续扩展

- 只有 repair 主张成立后才运行六个官方 EB-HAB subsets、共 300 tasks。
- Primary 目标是 Full 相对 S0 在 -3pp margin 下 non-inferior；superiority 为 secondary。
- 32B、EB-NAV/ALF、第二 teacher、rule growth、goal-model repair 和 graph-skill co-evolution全部延后。

## 1. Phase 3A：Evidence Reliability Guard（无人工）

### 1.1 研究问题

视觉 observation 与 simulator state 的 mismatch 在 partial observability 下不可完全消除。
本阶段不追求“视觉永远正确”，而检验一个小型、信息隔离的 Evidence Reliability Guard
是否能在保留足够 evidence coverage/recall 的同时，阻止不可靠观察继续传导成错误 mismatch、
归因、recurrence cluster 和 Skill proposal。

核心假设：

- H1：当前按最高 confidence 去重的融合会静默覆盖一部分跨来源 polarity conflict。
- H2：显式的 source-disagreement guard 可将 false contradiction 至少降低 50%。
- H3：该收益不能仅由提高 confidence threshold 解释。
- H4：Guard 不以 reject-all 获得表面安全；coverage >= 0.50，Skill-update recall 相对当前下降
  不超过 10 percentage points。
- H5：正式运行不增加视觉调用次数，token 成本不超过当前方案约 1.2x。

### 1.2 方法边界与信息隔离

Guard 位于 raw Evidence providers 和 ledger/mismatch 之间：

```text
feedback rule evidence --+
vision-only evidence -----+--> Evidence Reliability Guard --> accepted / unknown
episode-local ledger -----+
```

- Guard 不能读取 expected transition、active Skill、mismatch、attribution、patch、selection 或 audit。
- Simulator/PDDL state 只写独立 evaluation label，永不进入 `EvidenceRequest` 或 Guard。
- 第一版不实现主动重新观察、历史 transition retroactive rewrite 或新的 simulator action。
- 第一版不依赖 semantic/depth sensor；semantic visibility 仅作为可选 diagnostic smoke。
- `unknown` 不等价于 `false`；只有 Guard 接受的 evidence 才能写 ledger 和构造 mismatch。

### 1.3 对照臂

| Arm | 定义 |
|---|---|
| E0 Current | 当前 feedback + images/feedback VLM，按最高 confidence 去重 |
| E1 Feedback-only | 仅结构化 public feedback |
| E2 Threshold-only | 当前融合，但提高 visual confidence/coverage 门槛 |
| E3 Strict Guard | feedback 与 vision-only polarity conflict 一律降为 `unknown` |
| E4 Authority-aware Guard | 明确 action-local feedback 优先；视觉补未覆盖谓词；其余冲突为 `unknown` |
| E5 Oracle | evaluation-only simulator/PDDL truth ceiling |

E3/E4 正式运行只需要一次 vision-only call；实验期间为公平比较 E0 与 E3/E4，离线对同一图片
分别调用一次 images+feedback 和 images-only provider。Threshold/priority 参数只允许在 dev 与
selection 上选择；frozen audit 后不得调整。

### 1.4 自动 state oracle

通过外部 adapter 读取 EB-Habitat 内部 PDDL simulator state、goal expression 和 grasp manager：

- action 前后分别记录 queried predicates 的 truth；
- 重点覆盖 `holding/not_holding`、`on_top/at`、`open`、`near` 和 `task_complete`；
- oracle label 独立写入 artifact，并记录 task/episode/step/seed、predicate mapping provenance；
- 不修改 `EmbodiedBench/planner/` 或 Habitat 上游代码；
- 接口和 artifact 测试必须证明 oracle 字段不能被 evidence/model request 序列化。

P3A-0 Go 条件：holding/open/action-local mapping 接近 100%；全部 queried predicates mapping
coverage >= 0.90；相同 seed 下启用 evaluation label 不改变 RGB/action/environment result。

### 1.5 数据与划分

目标约 300 transitions：

- 200 natural transitions：正常冻结 Qwen executor rollout；
- 100 targeted stress transitions：补足罕见 action、success/failure、相似实例、视觉缺失和 no-op；
- nav/pick/place/open/close 每类尽量不少于 30 条；
- natural 与 stress 分开报告，combined 仅作 secondary；
- 120 dev / 80 selection / 100 frozen audit，按 task/scene 隔离；
- frozen audit definite contradiction/no-write cases 若少于 50，追加采集而不降低门槛。

若自然 executor 轨迹无法满足稀有 action 配额，允许使用固定、预记录的合法/非法 primitive action
脚本构造 stress set；该变更和每类实际样本量必须在查看 Guard outcome 前记录。

### 1.6 噪声压力测试

复用 `NoisyEvidenceProvider`，在相同 transition 上离线测试：

- visual polarity flip 10%；
- visual drop 20%；
- feedback polarity flip 10%；
- visual + feedback 各 10%；
- false-positive evidence 10%。

噪声结果作为 robustness diagnostic，不与 natural state-oracle 指标混为一个平均数。

### 1.7 指标

Evidence 层：predicate Macro-F1、contradiction precision/recall、false contradiction rate、coverage、
ECE、Brier、selective risk、source disagreement、unknown rate。

传导层：target/field Macro-F1、clean/control false `skill_update` rate、Skill-update precision/recall、
错误 recurrence cluster 数、proposal 数及 candidate audit class。

成本层：visual calls、prompt/completion tokens、tokens per accepted reliable predicate、离线重放时间、
live wall time/GPU hours。

### 1.8 预注册 Go/No-Go

离线 Go 需要同时满足：

1. false contradiction 相对 E0 下降至少 50%，paired bootstrap 95% CI 不跨 0；
2. contradiction precision >= 0.90；
3. selective coverage >= 0.50；
4. clean/control false `skill_update` rate < 5%；
5. Skill-update recall 相对 E0 下降不超过 10pp；
6. target/field Macro-F1 不低于 E0；
7. 正式运行 visual calls 不高于 E0，tokens <= 1.2x E0。

Decision：

- Go：冻结最佳 Guard，进入 live Habitat pilot；
- Partial Go：错误显著下降但 coverage/recall 不足，只允许研究 temporal confirmation，不跑主实验；
- No-Go：停止 Guard 扩展，保留诊断，不以增加 abstention 冒充可靠性提升。

### 1.9 分阶段执行与停止点

1. P3A-0：10-episode state-oracle feasibility smoke；失败则只修映射，不启动大规模调用。
2. P3A-1：采集/构建 300-transition 自动标注集；模型调用与 Habitat 串行分离。
3. P3A-2：所有 arms 在相同 transitions 上离线重放；dev/selection 选择并冻结 Guard。
4. P3A-3：只运行一次 frozen audit；按 1.8 决策。
5. P3A-4：仅离线 Go 后运行 `effect_pick_inversion` live pilot：Current vs Guard，seed 0，
   每臂 20 acquisition episodes，相同 executor/teacher/gate/audit budget。
6. P3A-5：effect pilot 通过后才复现 multihold 和 clean control；仍不直接扩到三 evolution seeds。

所有长任务必须在命名的 detached `tmux` 会话中运行，stdout/stderr 同时写入 `running/Phase3/phase3a/`
下的时间戳日志；命令、PID/tmux session、endpoint、config/manifest hash、开始/结束时间和退出码
写入 run manifest。已有输出禁止覆盖，中断恢复必须复用完整坐标并保留损坏/不完整 artifact。

## 2. 执行记录

### P3A-0 启动前状态（2026-08-24）

- 当前代码基线：commit `439aab7`（`Complete Phase 2 fault-repair validation`）。
- 工作区在 Phase 3 log 创建前为 clean。
- 现有测试：210 tests passing（`max_embench` 环境）。
- EB-Habitat 当前配置只启用 RGB 路径，`should_setup_semantic_ids: False`；核心实验不依赖 semantic。
- Habitat PDDL 已提供 `get_true_predicates/get_possible_predicates` 和 predicate `is_true`，具备自动
  state oracle 的实现基础。
- 当前 Evidence 去重按最高 confidence 选取；不同 confidence 的 polarity conflict 不会自动转为
  `unknown`，这是 H1 的直接实现动机。

后续每个运行和方案调整继续追加在本节，不回写 Phase 1/2 日志。

### P3A-0 实现与启动前验证（2026-08-24 21:35--21:50 CST）

- 新增外部只读 `HabitatStateOracle`，在 primitive action 前后查询 Habitat PDDL predicate，
  并以独立 `state_oracle_label` 行写入 artifact。Runner 先完成 Evidence/mismatch/attribution，
  再写 oracle label；oracle 不存在于 `EvidenceRequest.to_provider_payload()`。
- 新增 diagnostic-only CLI 开关 `--state-oracle-labels`；未显式使用 `--diagnostic` 时 fail closed。
- 新增第一版 `EvidenceReliabilityGuard`：Strict 和 Authority-aware 两种 mode；跨来源冲突、
  confidence/coverage 门槛以及每个决策的 provenance 均可审计。
- 将视觉 provider 拆成 `images+feedback` 与真正的 `images-only` prompt；后者不序列化 public
  feedback，并使用独立 usage purpose 计费。
- 启动前检查发现旧视觉 query 在首次 `pick` 时不保证查询 `holding/not_holding`，会造成
  oracle 有 action-local 标签而视觉臂无可比输出。因而在查看任何 Phase3A outcome 前，将
  `pick -> holding/not_holding`、`place -> not_holding` 加入 prediction-blind evidence query。
  这些 query 仅由已执行 action 及其参数生成，不读取 expected transition 或 Skill。
- 相关测试共 58 条通过：guard、visual prompt/query、oracle mapping/artifact isolation、runner
  和 CLI fail-closed wiring。下一步进入真实 Habitat 10-episode feasibility smoke。

### P3A-0 首次启动失败与调整（2026-08-24 21:47 CST）

- tmux session `vista_p3a0_20260824` 按要求 detached 启动，wrapper 和 child PID 分别为
  `3025034/3025038`；进程在创建 Habitat 和产生模型调用前 fail closed，exit code=1。
- 原因：使用 `configs/vista_p0.json`（预注册 evolution seeds `[0,1,2]`）却把 smoke 显式限制为
  seed `0`，触发 controlled-protocol seed 一致性检查。这不是 oracle、simulator 或模型故障，
  且没有产生不完整 experiment output。
- 调整：重试改用已经冻结为单 seed `[0]`、同一 executor/resolution/n-shots/manifest 的
  `configs/vista_fault_repair.json`，不注入 Skill fault。该配置差异只涉及 evolution/gate 参数；
  P3A-0 为 rule-only 数据采集，不执行 proposal/gate，因而不会改变 oracle feasibility 问题。
- 失败 artifact 保留在 `running/Phase3/phase3a/p3a0_oracle_smoke_20260824/`，不覆盖；重试使用新路径和
  新 tmux session。

### P3A-0 第二次启动失败与调整（2026-08-24 21:48 CST）

- `vista_p3a0_retry_20260824` 同样在 Habitat/model 调用前 fail closed，exit code=1。
- 原因：`configs/vista_fault_repair.json` 的冻结 manifest 是小型
  `configs/eb_hab_pilot_manifest.json`，与命令指定的正式 train-validation manifest 不同。
- 调整：改用 Phase2 已实际使用的 `configs/vista_fault_repair_fullsel_p10.json`；它同时固定
  train-validation manifest 与单 seed `[0]`。前两次错误均是 protocol preflight 捕获的配置
  组合错误，不计入实验 attempt，不删除 artifact，不改变预注册指标或阈值。

### P3A-0 运行中首个映射发现（2026-08-24 21:55 CST，outcome-blind 修复）

- 第三次启动 `vista_p3a0_run3_20260824` 已进入真实 Habitat（RTX 4090 EGL renderer），
  session、runtime manifest 和 JSONL 正常持续写入。
- 在只查看 oracle `mapped/unknown` 状态、不查看任何 Guard outcome 时，发现 Habitat 的零参数
  PDDL predicate 序列化为 `not_holding()`；通用 `PredicateKey.parse` 会把空括号误解析为空实体，
  使 `not_holding` 系统性标成 unknown。
- 已在 oracle adapter 内只对显式 `name()` 归一化为 `name`，并将 fake predicate 回归测试改为
  Habitat 的真实格式；9 条 oracle/runner 测试通过。正在运行的进程已加载旧代码，保留为失败
  定位样本；修复只对下一次复验生效。
- 对未匹配的 predicate 追加同族 PDDL candidates 到 rationale，便于下一次 smoke 区分“predicate
  family 不存在”和“实体归一化失败”，不改变 truth 判断。

### P3A-0 原始 smoke 结果与第二项映射修复（2026-08-24 22:03 CST）

- 原始 smoke 正常结束（exit code=0）：10 episodes、102 transitions、102 独立 oracle labels，
  transition/label coordinate linkage 100%，transition payload 中不存在 oracle 字段。
- 旧 parser 的 post-state 总 mapping coverage=`0.8814`，未达到预注册的 `>=0.90`；分谓词为
  holding=`1.00`、near=`1.00`、task_complete=`1.00`、not_holding=`0.00`、at=`0.00`、
  open=`0.50`。因此严格按 P3A-0 规则不能启动大规模采集。
- `not_holding` 已由零参数解析修复。进一步检查数据集可见的 public category 与 PDDL entity
  表示发现，VISTA query 使用 `ball/sponge/...`，PDDL concrete entity 使用 YCB instance name
  （如 `056_tennis_ball_:0000`），但 entity 的 `expr_type` 保存正确 public category。oracle 匹配
  因而新增“exact/category token 首选，PDDL entity type 回退”；同时把 public `refrigerator`
  与 PDDL `fridge` 归一化。该回退只决定 evaluation query 对应哪个 simulator predicate，不向
  Evidence/Guard 暴露任何值。
- 新增真实格式/type 回归测试后 oracle/runner 10 条测试通过。已启动
  `vista_p3a0_fixed_20260824` 做全 10 episodes 修复复验；未查看任何 Evidence Guard 指标。
- 原始分析 artifact：
  `running/Phase3/phase3a/p3a0_oracle_smoke_run3_20260824/oracle_feasibility_old_parser.json`。

### P3A-1 数据构造细化（冻结于 Guard outcome 之前，2026-08-24 22:10 CST）

- 200 个 natural samples 仍全部来自真实冻结 Qwen executor/Habitat transitions，且必须有独立
  post-state oracle label 和完整 pre/post RGB。
- 原计划笼统写作“100 targeted stress transitions”。为更直接检验 observation/state mismatch，
  将其具体化为 100 个**真实 transition 的确定性 observation stress variants**：中央遮挡、
  post blackout、identity blur、temporal duplicate（post=pre）和 feedback redaction 各约 20。
  每个 variant 继承同一 transition 的 simulator-state label，但不冒充新的独立环境 transition；
  natural 与 stress 始终分开报告。
- 该调整的原因是：用隐藏 simulator state 选择并执行合法/非法脚本会改变 action/state 分布，
  而确定性 observation perturbation 直接改变 Guard 可见信息、不改变 ground truth，更贴合本阶段
  的鲁棒性问题。若真实 natural source 中任一 nav/pick/place/open/close 仍少于 30，仍按原计划
  追加 scripted action collection，不能靠复制 stress variant 填“独立 transition”配额。
- 冻结数据量为 dev=`120`（80 natural+40 stress）、selection=`80`（53+27）、audit=`100`
  （67+33）；用 ReplicaCAD `scene_id` 做三路 scene-disjoint 分配，同一 source 与其 stress variant
  必须留在同一 split。audit 在参数冻结前不得打开。
- 新增 checkpoint/resume 的双视觉缓存：`images+feedback` 与真正 `images-only` 各对同一 300
  samples 调用一次；每条记录单独保存 error、attempt 和 token usage。新增自动多臂审计脚本，
  dev 筛参数、selection 确认、audit 只打开一次，并运行五种预注册噪声。

### P3A-1 stress 数据再调整（基于已有动作频率，仍在 Guard outcome 之前）

- 复核两个 Phase2 各约 20-episode 的自然轨迹后，动作计数分别为
  `{nav:80,pick:68,place:18,close:3,open:2}` 和
  `{nav:88,pick:73,place:11,close:3,open:1}`。因此单靠 40–60 natural episodes 几乎必然
  无法让 open/close 达到 30 个独立 transitions。
- 据此正式启用 1.5 中已预注册的 fixed primitive script 条款，300 主数据恢复为
  **200 个真实 natural + 100 个真实 scripted stress transitions**；上一节定义的 observation
  perturbation 不进入 300 主数据，仅由 frozen audit 的 drop/flip/false-positive 噪声诊断覆盖。
- Script 不读取 simulator truth 或运行结果来选下一动作，只按 public action catalog 固定执行：
  invalid place、invalid pick、未导航 open、nav、open、reopen、close、reclose。使用 episodes
  40–59，与 natural episodes 0–39 隔离；每一步仍有真实 RGB/feedback/state-oracle label。
- 数据 builder 对 stress split 预留 open/close 最低数，最终 combined dataset 对
  nav/pick/place/open/close 每类都执行 `>=30` 的硬检查；不满足则停止，不启动 VLM 缓存。

### P3A-2 传导指标真阳性构造（冻结于 Guard outcome 之前）

- Clean shared Skill 的自然 transitions 几乎没有真实 `skill_update`，只用它们无法计算
  “错误 update 被压低的同时，正确 update recall 是否保留”。因此每个 pick transition 除 clean
  case 外，离线增加两个**只修改 expected branch 的 counterfactual fault cases**：
  `effect_pick_inversion` 和 `constraint_pick_multihold`；observed RGB/feedback/oracle label 完全不变。
- Gold routing 由同一 `CreditAssigner` 在 simulator-state oracle evidence 上计算；这是自动机制标签，
  不冒充人工意图 gold。Clean case 用于 false `skill_update`，counterfactual cases 为 recall/field
  F1 提供真阳性。`task_complete` 从该 action-local 传导指标中排除，因为 rule-only collector
  没有 learned goal grounding；包含 task_complete 的 predicate 指标仍单独报告。
- Paired bootstrap 以 source transition 为重采样单元，将同一 source 的 clean/effect/constraint
  cases 作为一个 cluster，避免把三个 counterfactual rows 错当独立样本。

### P3A-2 H3 冻结规则澄清（任何 VLM outcome 之前）

- E2 threshold-only 是解释性对照，不能作为“最佳 Guard”被冻结；deployable candidates 只允许
  E3 Strict 或 E4 Authority-aware。
- Guard 在 dev/selection 冻结参数后，audit 同时评估使用完全相同 confidence/coverage 参数的
  E2 matched-threshold arm。
- 最终 Go 新增 H3 硬条件：Guard 的 false-contradiction rate 必须严格低于 matched E2，且
  predicate coverage 不得比 matched E2 再下降超过 10pp。否则即使相对 E0 有提升，也判断为
  “阈值已足够解释”，不进入 live Guard pilot。
- 参数网格在任何 Phase3A VLM 调用前冻结为 confidence=`{0.50,0.60,0.75,0.90}`、
  coverage=`{0.00,0.25,0.50,0.75}`。纳入 coverage=0 是因为 Phase2 真实 artifact 已观察到
  Qwen/vLLM 会把若干有明确 rationale 的 coverage 输出为边界附近 `1e-6`；把最小 coverage
  预先截在 0.25 会把“coverage score 是否有用”和“跨来源冲突策略是否有用”混为一谈。
  `coverage>=0.50` 的系统级 Go 门槛保持不变，低阈值不能靠 reject-all/accept-all 自动过关。
- 若没有任何 E3/E4 candidate 同时通过 dev 与 selection gates，分析脚本直接返回
  `no_go_no_candidate_passed_dev_and_selection`，不计算 audit oracle metrics，并将
  `audit_opened_once=false`；不能为了“完成表格”提前消费 frozen audit。

### P3A-1 双 endpoint 交叉平衡（任何 Phase3A VLM outcome 之前）

- 为避免把两个 vLLM server 的潜在配置/吞吐差异与 evidence condition 混淆，两个视觉条件都
  不固定绑定单一 endpoint。
- 按冻结 dataset index 在 `192.168.1.185:8000` 与 `192.168.1.173:8001` 间 round-robin；
  images-only 使用 `+1` offset，所以对同一样本两个 condition 落到相反 endpoint。每个 condition
  在两个服务上各约 150 calls，每条 cache row 记录实际 `base_url` 和独立 token usage。
- 两端已分别通过同一 6/6 contract probe，model id 均为冻结
  `Qwen/Qwen3-VL-8B-Instruct`。

### P3A-0 中间复验结果（2026-08-24 22:06 CST）

- type-fallback 版 10 episodes 再次得到 102/102 transition-label linkage，整体 post mapping
  coverage 从 `0.8814` 提升至 `0.9888`；holding/not_holding/near/open/task_complete 全为
  `1.00`，但 7 个 `at` 仍全部 unknown，导致 place action coverage=`0.8727`。
- 新 rationale 证明 `get_possible_predicates()` 根本没有返回任何 `at/on_top/in` 候选，而非
  category/type 仍匹配失败。检查 Habitat 上游实现发现 possible 使用实体 combinations，true
  predicates 使用 permutations；二元谓词的有序实体组合因此可能漏掉。
- 最终 oracle 改为：先匹配 `get_true_predicates()` 产生正例；若无正例、但 predicate family
  存在且每个 query entity 能在 PDDL domain 按 exact/category/type 解析，则按 domain
  closed-world 标 false；无法解析仍为 unknown。新增 missing-permutation 回归测试；随后加入
  live Guard wiring 测试后总测试 226 条通过。
- 已在 `vista_p3a0_final_20260824` 中启动最终 10-episode smoke。中间 artifact 保留为
  `running/Phase3/phase3a/p3a0_oracle_fixed_20260824/oracle_feasibility_type_fallback.json`。

### P3A-0 最终 mapping 结果（2026-08-24 22:13 CST）

- 最终 10 episodes/102 transitions 正常结束，102/102 coordinate linkage；post oracle 共
  624 queries，mapped=`624`、unknown=`0`、coverage=`1.000`。
- 分谓词 coverage：holding/not_holding/at/near/open/task_complete 全为 `1.000`；分 action 的
  nav/pick/place/close 也全为 `1.000`（本批没有 open action，但 open predicate 4/4 已映射）。
- transition payload 扫描确认不包含 `state_oracle` 或 evaluation-only source，信息隔离通过。
- Artifact：`running/Phase3/phase3a/p3a0_oracle_final_20260824/oracle_feasibility_mapping.json`。
- 已启动同 seed、同 10 episodes、关闭 oracle 的 `vista_p3a0_reference_20260824`；待比较 action、
  feedback、success、pre/post image SHA 和 episode outcome 的 exact signature。

### P3A-0 Go（2026-08-24 22:20 CST）

- Oracle-on 与 oracle-off 均完成 10 episodes/102 transitions。
- 两个运行的 action payload、last-action success、feedback、pre/post image SHA256、episode
  trajectory、task success/progress、environment steps 和 invalid-action count exact match=`true`。
- 因此 P3A-0 同时通过 mapping coverage、artifact isolation 和 rollout non-interference 三项门槛，
  允许进入 P3A-1。最终 artifact：
  `running/Phase3/phase3a/p3a0_oracle_final_20260824/oracle_feasibility_with_reference.json`。
- 两个远端冻结 Qwen3-VL-8B endpoints（`192.168.1.185:8000`、
  `192.168.1.173:8001`）均通过仓库规定的 6/6 serving-contract probe。

### P3A-1 Natural collection 完成（2026-08-24 22:34 CST）

- `vista_p3a1_natural40_20260824` 正常结束：40 episodes、363 transitions、363 oracle labels，
  exit code=0；scene transition capacities 为 `[49,78,45,68,72,51]`，足以构造 200 条
  scene-disjoint natural 子集。
- 动作计数：nav=`179`、pick=`144`、place=`17`、open=`16`、close=`7`；再次确认真实
  place/open/close 需要 scripted stress 补足。
- 2089 个 post oracle queries 中 definite=`2080`、unknown=`9`，coverage=`0.9957`，高于
  P3A-0 门槛。9 个 unknown 全来自 episode 34 的 feedback parser 额外生成的 lossy alias
  `open(cabinet_push)`；同一 action-local oracle query 保留带编号 `cab_push_point_i` 并可映射。
  Audit 的 predicate gold 本来就跳过 oracle unknown；传播 oracle evidence 也冻结为只使用 definite
  state，不把 unknown 当负例或可更新证据。

### P3A-1 Scripted stress collection 完成（2026-08-24）

- `vista_p3a1_stress_20260824` 在 detached tmux 中正常结束，exit code=0；共采集
  episodes 40–59 的 20 episodes、155 个真实 simulator transitions，因 2 个 episode 提前
  task complete，略少于固定日程上限 160。
- 动作计数：place=`20`、pick=`20`、open=`58`、nav=`20`、close=`37`。成功/失败同时
  被覆盖：place 0/20、pick 1/19、nav 20/0、open 19/39、close 19/18，因此它确实增加了
  feedback/state 不一致的压力样本，而不是复制图片。
- 753 个 post oracle queries 中 definite=`691`、unknown=`62`。unknown 仍全部是 feedback
  parser 的 lossy alias `open(cabinet_push)`，具体 action-local `cab_push_point_i` 查询仍为
  definite；冻结 audit 规则忽略 unknown，不伪造负标签。
- Artifact：`running/Phase3/phase3a/p3a1_scripted_stress_20260824/events.jsonl`、
  `running/Phase3/phase3a/p3a1_scripted_stress_20260824/runtime_manifest.json` 和
  `running/Phase3/phase3a/p3a1_scripted_stress_20260824/collection_manifest.json`。

### P3A-1 首次 dataset freeze 失败与预注册修复（任何 VLM outcome 之前）

- 首次 builder 在写出数据前触发硬门槛：300 条中 action 计数为 nav=`89`、pick=`89`、
  place=`26`、open=`55`、close=`41`，place 还差 4 条。本次失败没有生成 dataset，也没有
  发起任何 VLM 请求。
- 原因是 natural 子集已经取全 17 条 place，而原 stress 抽样只硬保留 open/close，
  round-robin 总共只留 9 条 place。这是 stratified selector 的配额缺口，不是收集数据
  不足。
- 在没有观察任何 Guard/VLM 结果的前提下，将 stress 的 place 最低配额冻结为
  dev/selection/audit=`5/4/5`；总 stress place 至少 14，与natural 的 17 合计至少 31。
  这只修复早已声明的“五类动作每类 >=30”硬约束；300 总量、120/80/100 split、
  scene-disjoint 和 open/close 最低配额保持不变。
- 修正后 builder 成功冻结 `vista_phase3a_evidence_state_oracle_v2`：300 条，
  dev/selection/audit=`120/80/100`，natural/stress=`200/100`，动作计数为
  nav=`87`、pick=`86`、place=`35`、open=`53`、close=`39`，三个 split 之间无 scene
  overlap。Artifact：`running/Phase3/phase3a/phase3a_dataset_v2_20260824.json`。

### P3A-2 VLM cache 启动前的数据外发安全门（2026-08-24）

- 按冻结设计尝试启动两个 detached tmux cache 任务时，安全审批在进程创建前拒绝：
  300 条 RGB、environment feedback 和派生 evidence query 将被发送到
  `192.168.1.185:8000` 和 `192.168.1.173:8001`，需要用户对这一具体数据外发明示
  知情授权。
- 本次拒绝发生在任何 tmux/VLM 进程启动之前，因此没有发送样本，也没有生成
  部分 cache。检查本机 `127.0.0.1:8000` 没有可用服务，暂无不外发的等价替代。
- 下一步只能在用户明确同意将上述 Phase3A 数据发送到这两个 LAN vLLM endpoints 后
  继续；不会通过其他命令或间接路径绕过该门槛。
- 2026-08-25，用户明确同意按现有规划完成 Phase3A，包括将这些模拟器 RGB、
  环境反馈和派生查询发送到上述两个 LAN vLLM endpoints。据此解除本安全门，
  恢复 P3A-2；实验设计、dataset 和 threshold grid 均不因授权事件改变。

### P3A-2 VLM evidence cache 完成（2026-08-25）

- 两个 detached tmux 任务均正常结束；`images_and_feedback` 与 `images_only` 各 300 条，
  sample id 唯一、与冻结 dataset 顺序 exact match，error=`0`、retry=`0`。两个 condition 都在
  两个 endpoint 上各 150 calls，crossover 平衡达成。
- `images_and_feedback` 有 5 条空 evidence，prompt/completion tokens=`230253/109199`；
  `images_only` 有 11 条空 evidence，prompt/completion tokens=`223211/126112`。空 evidence
  是模型的有效覆盖结果，不重跑、不删样本。
- Cache SHA256：images+feedback=`80babf32b4bee8b2d5ef9db9a70ce7b7102ece3684402c3a1b74d0145639e70c`；
  images-only=`86f53ef238e6ef6e873c5a8445b1a58782907b70158911ff09f392209b22ae09`。
- Artifacts：`running/Phase3/phase3a/cache_images_feedback_20260824.jsonl`、
  `running/Phase3/phase3a/cache_images_only_20260824.jsonl` 及各自 `.summary.json`；运行记录在
  `running/Phase3/phase3a/p3a2_cache_images_feedback_20260824/` 和
  `running/Phase3/phase3a/p3a2_cache_images_only_20260824/`。

### 首次 frozen audit 中止与 Oracle cache 缺陷（2026-08-25）

- 首次 audit 程序正常结束为 `no_go_no_candidate_passed_dev_and_selection`，32 个 E3/E4
  candidate 无一通过 dev gates；程序按预注册规则没有计算 audit metrics，
  `audit_opened_once=false`。但机制 sanity check 同时发现，原本应由 pick counterfactual
  产生的 `skill_update_gold_count` 在 dev 为 0，因此不能把这个 No-Go 解读为 Guard 失败。
- 逐条检查发现 Oracle 存在逻辑不可能的 post-state：成功 pick 后同时
  `holding(x)=true` 与 `not_holding=true`；成功 close 后 `open=true`。根因是
  EB-Habitat 在 `task.step` 开始时清空 PDDL `pred_truth_cache`，随后 action precondition
  检查重新写入动作前真值，但 apply action 之后不再清空。旧 Oracle 在 post observe
  中因此混合了动作前缓存和动作后新计算值。
- 修复为：每次 evaluation-only `observe` 前调用 `sim_info.reset_pred_truth_cache()`，只清除
  谓词求值缓存，不修改 simulator state。新增 regression test 确保 reset 先于
  `get_true_predicates()`。由于旧 label 不再可作 truth，接下来必须重收 Oracle labels。
- 第一次输出保留为 `running/Phase3/phase3a/phase3a_guard_frozen_audit_20260825.json`，用作
  failure-analysis artifact，不作为 Phase3A 方法结论。因 `audit_opened_once=false`，修正 label 后仍可
  严格执行一次有效 frozen audit。

### Oracle cache-fix smoke 通过（2026-08-25）

- 修复后重跑同 seed 10 episodes：102 transitions/102 labels，624/624 post queries mapped，
  coverage=`1.000`；Oracle 仍未进入 transition payload。
- 新增逻辑一致性检查：13 个存在 `holding(x)=true` 的 post-state 全部为
  `not_holding=false`；52 个无 holding 的可查询 post-state 全部为 `not_holding=true`；
  成功 close 后 open 查询全为 false，violation=`0`。
- 与原 oracle-off reference 的 action、feedback、success、image hash、trajectory/outcome 对比
  仍为 102/102 exact match，证明清理 evaluation cache 不干扰 rollout。Artifact：
  `running/Phase3/phase3a/p3a0_oracle_cachefix_20260825/oracle_feasibility_with_reference.json`。
- 据此 cache-fix 通过 P3A-0，允许重收 P3A-1 labels；为避免两个 Habitat 实例资源干扰，
  先收 20 stress episodes，再收 40 natural episodes，不并行。

### P3A-1 cache-fix label 重收完成（2026-08-25）

- Stress 重收：20 episodes、155 transitions/155 labels，holding/not_holding 逻辑冲突=`0`，
  成功 open/close 的 action-local state 方向冲突=`0`。与旧 stress 的 coordinate、action、
  feedback、success 和 pre/post image SHA 全部 exact match。
- Natural 重收：40 episodes、363 transitions/363 labels，holding/not_holding 逻辑冲突=`0`。
  Oracle post values 为 true=`882`、false=`1198`、unknown=`9`；9 个 unknown 仍仅来自
  lossy `open(cabinet_push)` alias。与旧 natural 的 363 个 coordinate 全部一致，VLM
  request 签名（action、feedback、success、instruction、pre-ledger、goals、两帧 image SHA）差异=`0`。
- 因此旧新 518 个 source transitions 的 VLM 输入字节与结构等价，只有 evaluation-only
  oracle label 被修正。复用已冻结的 600 条 cache 不是选择性重采样，无需再花费 VLM
  calls；任何一条签名不等时原计划会强制重跑，但本次为 0 差异。
- 新主数据版本标识冻结为 `vista_phase3a_evidence_state_oracle_cachefix_v3`；v2 保留仅作
  Oracle cache bug 的 failure-analysis artifact。

### Guard v1 No-Go 与 Guard v2 dev-only 修订（2026-08-25）

- 修正 Oracle 后的 Guard v1 再次在 dev 阶段停止：32 个 images-only E3/E4 candidates 均未
  通过，`audit_opened_once=false`。E0 dev false-contradiction rate=`0.0063`，而原始
  images-only Authority 的假矛盾率=`0.0696`、错误 Skill update=`0.056`。
- Dev row-level 归因表明 22 个 false contradictions 中：12 个是 action 明确失败后的
  visual-only holding/not_holding 幻觉，10 个是 visual-only `at=false`；这些输出的
  confidence/coverage 均接近 1，证明只调数值阈值不能解决。
- 因此只使用 dev 冻结 Guard v2 的三项 prediction-blind 规则：
  1) failed high-level action 无结构化支持时拒绝 visual-only state claim，并对未被 feedback
  更新的 definite pre-ledger 做 no-op persistence；2) `at=false` 需要结构化 feedback；
  3) visual-only 只接受动作局部谓词（nav→near、pick→holding、place→at/holding、
  open/close→open）。成功 place 还用 action target + definite pre-action held object 产生
  action-local `at=true`，不读取 Skill/expected delta。
- 纯 images-only 条件在 dev 上的导航关系语义质量低，即使加上上述规则仍会使
  target macro-F1 低于 E0。因此部署候选改为与 E0 相同的**单次
  images+feedback VLM extraction**，再与确定性 feedback parser 分支做 Guard 融合；images-only
  cache 保留为独立性消融。这不增加 VLM call，且 matched E2 也用同一 cache。
- 在未查看 selection/audit 的情况下，该 Authority Guard dev 达到 coverage=`0.578`、
  false-contradiction=`0`、clean false Skill update=`0`、Skill recall=`1.0`、target macro-F1=`0.792`，
  高于 E0 的 `0.776`；允许将 Guard v2 冻结后进入 selection。

### Guard v2 selection、唯一一次 audit 与最终 No-Go（2026-08-25）

- Dev 上 32/32 个 E3/E4 candidates 通过安全/覆盖/传导门槛；selection 上 32/32
  再次确认。按预注册 rank 冻结 `E4_authority:0.50:0.00`，matched threshold 为
  `E2_threshold:0.50:0.00`。此时首次且唯一一次打开 100 条 audit，
  `audit_opened_once=true`；打开后不再修改 Guard 或阈值。
- Audit E0 → Guard v2：false-contradiction rate `0.00738 → 0.00000`（relative reduction=100%）；
  contradiction precision `0.9893 → 1.0000`；clean false Skill update 均为 `0`；Skill update
  recall 均为 `1.000`；target macro-F1 `0.7484 → 0.7578`；field macro-F1 均为 `1.000`。
- Guard 的代价是更保守：predicate coverage `0.7475 → 0.5990`，predicate macro-F1
  `0.6433 → 0.5789`，contradiction recall `0.8685 → 0.1080`。Matched E2 coverage=`0.7475`，
  Guard 比它低 `0.1485`，超过 H3 允许的 10pp；因此无法证明收益不是主要来自
  “拒绝更多证据”。
- Paired cluster bootstrap 的 absolute reduction=`0.00738`，95% CI=`[0.00000, 0.02479]`；
  下界没有严格大于 0。因 E0 audit false positives 本身极少，当前 100 条 audit 不足以
  给出稳健非零改善证据。
- 成本实际相同：E0 和 Guard 都是 100 calls、prompt=`75703`、completion=`43590`。
  审计 JSON 的 serializer 初版误把 images-only cache 数字写入 `audit_cost.guard`，但 Go 计算时
  已正确使用 fb cache，两个 cost checks 原本就是 true。原 artifact 不修改，报告性更正保存在
  `running/Phase3/phase3a/phase3a_guard_v2_frozen_audit_v4_20260825.cost_amendment.json`，对决策无影响。
- Frozen noise audit：visual flip 10%、visual drop 20% 和 false-positive 10% 下 Guard 仍为
  false-contradiction=`0`、false Skill update=`0`、Skill recall=`1.0`；但 feedback flip 10%
  时 false-contradiction=`0.1107`、contradiction precision=`0.40`、Skill recall=`0.7143`。说明 v2
  对视觉噪声稳健，但 authority policy 对结构化 feedback 错误过于敏感。
- 最终 `decision=no_go`：11 个 Go checks 中只有 `paired_ci_excludes_zero` 和
  `guard_gain_not_explained_by_matched_threshold` 失败。依预注册规则，不启动 P3A-4 live pilot，
  也不执行条件性 P3A-5 扩展；这两项是“被门槛取消”而不是未完成。
- 最终 artifact：`running/Phase3/phase3a/phase3a_guard_v2_frozen_audit_v4_20260825.json`；
  第一次无效 No-Go 和 Guard v1 No-Go 仍保留作 failure-analysis，不用于最终方法声称。

## 2. Phase 3B：Feedback 依赖与短时视觉历史（预注册于 2026-08-25，任何 Phase3B outcome 之前）

### 2.1 研究问题与能力边界

Phase3A 证明当前 authority-aware Guard 能消除该 frozen audit 上的 false contradiction，
但它以 14.85pp coverage 损失换取安全，并在 feedback polarity flip=10% 时明显退化。因此
Phase3B 不再继续调 Guard 阈值，而回答两个彼此分离的问题：

1. 冻结 Qwen3-VL-8B executor 在 EB-Habitat 中拿掉 public environment feedback 后，短时 RGB
   历史能否补回规划表现；
2. evidence branch 在完全不读取 feedback 的条件下，多一帧历史和最近动作能否显著优于当前
   two-frame images-only provider，并达到足以支撑持久 Skill 更新的精度。

本阶段仍固定 executor/teacher=`Qwen/Qwen3-VL-8B-Instruct`、temperature=0、request seed=0、
10-shot、resolution=500、max completion=4096。它是 frozen 8B 的小规模机制验证，不外推为
Claude/32B 或经过 WAP 式大规模训练后的能力。Simulator state 只用于 evaluation label；所有
no-feedback arm 的 model request 禁止序列化 feedback、expected transition、Skill mismatch、
attribution 或 oracle state。

### 2.2 P3B-1 executor 四臂 paired pilot

在 manifest acquisition role 的前 30 个固定 episode 上，使用相同 S0、相同 task 顺序、seed、
budget 和 endpoint，禁用 evolution 以及 evidence-ledger 写入，只改变 executor prompt 中的
feedback 和 RGB history：

| Arm | 当前 RGB | 最近动作历史 | environment feedback | RGB history |
|---|---:|---:|---:|---:|
| C0 current | 是 | 是 | 是 | 1 frame |
| C1 no-feedback | 是 | 是 | 否 | 1 frame |
| C2 temporal no-feedback | 是 | 是 | 否 | latest 3 frames |
| C3 temporal + feedback | 是 | 是 | 是 | latest 3 frames |

所有臂只注入 frozen S0 文本，不运行 VTCA/evidence/teacher，以免 feedback 通过 compact ledger
侧漏到 C1/C2。主指标为 task progress；同时报告 success、invalid-action rate、adjacent action
repetition rate、planner error、executor calls/tokens 和 wall time。repetition 只是 8B 卡住/循环的
诊断，不将合法的连续 navigation 自动当错误。

P3B-1 screening Go 需同时满足：C2 mean progress 严格高于 C1；C2 不比 C0 低超过 0.10；
C2 invalid/repetition rate 各不超过 `max(1.25 * C0, C0 + 0.02)`。paired task bootstrap 95% CI
照常报告，但 30-task pilot 不要求 CI 排除 0。C3 用于区分“历史本身有害”与“历史只有和
feedback 配合才有效”，不参与选择性调参。

### 2.3 P3B-2 no-feedback temporal evidence

只使用 Phase3A frozen dataset 的 dev=120 和 selection=80，不读取已经消费过的 audit=100：

- T0 pair-only：复用冻结的 images-only two-frame cache，不重新调用模型；
- T1 temporal-safe：按时间顺序提供 previous-pre/current-pre/current-post（episode 起始最多两帧）
  和最近至多 4 个 public action；不提供任何 feedback；固定一次新 Qwen call/sample；
- 每个 observation 必须输出 evidence tier=`direct_current|temporal|action_prior|unknown`；只有
  `direct_current` 和 `temporal` 可进入持久 mismatch/Skill-update 审计，`action_prior` 只能作为
  episode 内弱先验，不能授权 long-term write；
- confidence 固定最低 0.50，不进行 threshold grid。帧和动作历史从原始 transition artifact
  按 episode/step 构造，禁止从 oracle outcome 选择帧。

先看 dev；只有 dev 同时通过以下条件才打开 selection：T1 predicate Macro-F1 至少比 T0
+0.10、accepted predicate precision>=0.90、query coverage>=0.50、contradiction recall>=0.50、
clean false Skill-update rate<5%、target Macro-F1 不低于 matched E0（feedback-conditioned current）
超过 0.05，且 tokens<=1.5x T0。Selection 必须原样复验全部门槛；任何一项失败即 No-Go，
不修改 prompt/tier/阈值后重看 selection。

这里的 query coverage 是 simulator-state 可判定的 action-local query coverage，不冒充人工
`visual inferability` 标签。T1 若通过，只能说明短时历史在这个自动 oracle 上具备可用增量；
仍不能说明遮挡状态都能靠 RGB 恢复。

### 2.4 P3B-3 条件性 fresh audit 与 late feedback fusion

只有 P3B-1 与 P3B-2 都 Go，才收集未用于 Phase3A 的 episode 60--79（state oracle 仅作标签）
并进行一次 fresh audit。Fresh audit 前冻结 temporal provider；不得使用 Phase3A audit 调参。
Fresh audit 继续要求 P3B-2 全部门槛，并以 scene/task overlap=0 为硬检查。

Fresh temporal audit 通过后，才允许实现 late feedback fusion：视觉/temporal 为主证据，
feedback 只在 action-local、一致时提高 episode control；发生 polarity conflict 时输出 unknown/
pending。Feedback-only claim 可以帮助当前 episode replan，但单独不能授权 persistent Skill write。
该 fusion 还必须在 feedback flip=10% stress 下不劣于 pure temporal-safe，才允许进入 VTCA v2。

若任一前置 gate 失败，后续 fresh audit/fusion 记为“被预注册停止条件取消”，而不是缺失实验；
结论应收缩为失败发生的具体层级。全部长任务在 detached tmux 中运行，runtime manifest、命令、
endpoint、exit code、artifact hash、异常与因果调整均追加记录，不覆盖 Phase3A artifact。

### P3B-2 启动前协议修订：严格 no-feedback matched baseline（任何 Phase3B VLM outcome 之前）

- 检查 Phase3A cache builder 后发现：`images_only` 虽然删掉了 raw feedback 字符串，但仍把
  episode-local `pre_ledger` 的 predicate/value 放入 prompt；该 ledger 在原 rollout 中主要由
  feedback parser 写入。因此它是“无 feedback 文本”，不是严格 information-isolated
  no-feedback，不能作为 P3B 的 T0。
- P3B 改为重新缓存两个 matched 条件，各 dev+selection=200 calls：T0-strict 只看 current
  pre/post、current action 和 action-local query；T1-temporal-strict 增加 previous-pre 和最近
  至多 4 个 action。两者都不序列化 feedback、last_action_success、pre-ledger value、expected、
  oracle、mismatch 或 Skill。
- place 的 `at(object,receptacle)` query 只允许从最近 public pick action 的 object 参数构造；
  这只是待观察问题，不把 pick 是否成功作为事实。T0 没有历史 action，因此只查询
  `not_holding`；T1 可额外查询 `at`。模型输出被限制为请求内 predicates。
- 原文“复用 Phase3A images-only cache”的成本优势取消；原 cache 仅保留为 leakage diagnostic，
  不参与 P3B Go/No-Go。T1 的 `tokens<=1.5x T0` 现在比较两个新 cache，保持 matched。

### Phase3B 启动安全门（2026-08-25）

- 预注册、四臂 executor runner、strict pair/temporal cache、gated analyzer 和 tmux orchestration
  已实现；仓库完整测试套件 232 条通过。两个目标 endpoint 的 `/v1/models` 只读探测均在线。
- detached tmux 创建在任何实验 child process 启动前被安全审批拒绝。原因是 P3B 会把新的
  simulator RGB、instruction、public action history（C0/C3 还含 environment feedback）发送到
  `192.168.1.185:8000/v1`，并把 strict evidence 的 RGB/action/query 交叉平衡发送到该 endpoint
  与 `192.168.1.173:8001/v1`；Phase3A 的外发同意不自动覆盖这批新 payload。
- 因拒绝发生在 `tmux new-session` 之前，Phase3B model call=`0`、新 simulator episode=`0`，
  没有部分结果或 outcome 泄漏，预注册仍有效。只有用户明确同意上述 P3B 数据和两个 LAN
  destination 后才能解除此门槛；不得用间接命令绕过。
- 2026-08-25，用户在确认“客户端保留原文件、LAN server 只接收推理请求中的图像/文本数据”
  后明确回复“可以，开始Phase3B的实验验证”。据此授权 P3B 新 simulator RGB、instruction、
  action history、C0/C3 feedback 和 strict evidence queries 发送到上述两个 LAN endpoints；
  安全门解除。实验 protocol、样本、阈值与停止条件不因授权事件改变。

### P3B 运行中模型截断与 fail-closed 恢复（2026-08-25）

- T1 sample 135（`natural:21:s15`，endpoint `192.168.1.173:8001`）连续 3 次生成到
  4096 completion-token 上限并产生 truncated JSON；三次 calls/tokens 全部保留，最终 row
  记录 `error=JSONDecodeError`、evidence 为空。之后 endpoint 健康检查在线，sample 136 起正常推进。
- 该失败不补采、不删除、不换 endpoint，按系统既有“malformed evidence => no evidence => abstain”
  contract 计分。Cache driver 原先在 200 rows 完整但存在 error 时返回 exit 2，这会阻止只读
  analyzer；在不修改已生成 row、evidence 或 cost 的前提下，将 process-level 成功条件修正为
  “200 coordinates 完整”，error_count 继续作为质量指标。当前已加载的 child 仍会按旧代码在
  阶段末 exit 2，因此预计 tmux wrapper 停止；届时从已完成、hash 固定的 cache 启动 analyzer，
  不重新运行任何 model sample。

### P3B-1 executor 四臂结果（2026-08-25）

- 全部四臂在 detached tmux 中完成相同 episode 0--29，episode id/order exact match，四个
  events SHA 均通过 finalizer 复验；planner output error 均为 0。C0/C1/C2/C3 分别用时
  677/497/564/588 秒。
- 结果如下；invalid 以 environment step 为分母，repetition 以相邻 action opportunity 为分母：

| Arm | progress | success | invalid rate | adjacent repetition | calls | total tokens |
|---|---:|---:|---:|---:|---:|---:|
| C0 current | 0.6389 | 0.6000 | 0.3701 | 0.0159 (4/252) | 129 | 635,591 |
| C1 no-feedback | 0.5913 | 0.5333 | 0.3515 | 0.0291 (5/172) | 103 | 485,509 |
| C2 temporal no-feedback | 0.6347 | 0.6000 | 0.3879 | 0.0495 (10/202) | 118 | 602,713 |
| C3 temporal + feedback | 0.7597 | 0.6667 | 0.3360 | 0.0362 (8/221) | 112 | 589,837 |

- C2-C1 progress=`+0.04345`，paired task bootstrap 95% CI=`[-0.0667, 0.1536]`；
  C2-C0=`-0.00417`，CI=`[-0.1667, 0.1625]`。因此短时 RGB 在均值上补回了去掉 feedback 的
  损失，但 30-task CI 很宽，不能声称显著替代。
- 四个 screening checks 中，C2>C1、C2 距 C0 不超过 0.10、invalid<=limit 三项通过；
  repetition limit=`0.03587`，C2=`0.04950`，唯一失败。因此 P3B-1 按预注册为 `no_go`。
- Repetition failure 是窄门槛而非“10 次都必然错误”：C2 的重复分布在 6 个 episodes，其中
  episode 3 与 22 最终成功；但 triple-loop 仍由 C0 的 1 增至 C2 的 2。当前 metric 会计入
  合法连续动作，故只能把它解释成冻结 8B 的循环风险信号，不能事后换指标把 No-Go 改为 Go。
- C3-C0 progress=`+0.12083`，CI=`[-0.0250, 0.2667]`；说明 extra RGB history 在保留 feedback
  时至少没有整体破坏 performance，并可能有互补性，但 CI 跨 0，且 C3 不参与 gate，不能据此
  越过 P3B-1 的 no-feedback 目标。

### P3B-2 strict no-feedback cache 与信息隔离（2026-08-25）

- T0/T1 都完整覆盖 frozen dev+selection 200 个相同 coordinates，顺序 exact match；Phase3A
  audit rows 未进入 cache。T0 200 calls、error=0、tokens=`171,660`；T1 因 sample 135 的三次
  truncated attempts 共 202 calls、error=1、tokens=`264,919`。该 error row 保留为空 evidence。
- Cache-level raw accepted evidence 为 T0=`159`、T1=`163`；计分前按 predicate key 去重，同一
  structured response 内的重复项不增加 coverage。T1 全部 tier counts 为 direct=`246`、
  temporal=`6`、action_prior=`34`、unknown=`242`；action_prior 全部拒绝 persistent write。
- 两个 cache 的 isolation audit 均为 true：没有序列化 feedback、last-action success、pre-ledger、
  expected、oracle 或 Skill；frame SHA/query/action-count/provenance 均保留。Finalizer 重新验证了
  200 unique ids、cache hash 和同序配对。

### P3B-2 dev gate 与最终 No-Go（2026-08-25）

- Frozen analyzer 只计算 dev=120；由于 dev 未过，selection oracle metrics 未计算，
  `selection_opened=false`。Phase3A audit samples read=`0`。
- 主要 dev 结果：

| Arm | predicate F1 | asserted precision | coverage | contradiction precision | contradiction recall | false Skill update | Skill recall | target F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| T0 strict pair | 0.1523 | 0.9787 | 0.0882 | 0.9545 | 0.3302 | 0.0000 | 0.0000 | 0.3044 |
| T1 strict temporal | 0.0966 | 0.9655 | 0.0544 | 0.7143 | 0.0157 | 0.0000 | 0.0000 | 0.3044 |
| E0 feedback-conditioned current | 0.7061 | 0.8127 | 0.7711 | 0.9927 | 0.8553 | 0.0000 | 1.0000 | 0.7764 |

- T1-T0 predicate F1=`-0.0557`，与要求的 `>=+0.10` 方向相反；coverage 只有 5.44%，
  contradiction recall 只有 1.57%，target F1 比 E0 低 0.472。7 个 Go checks 只通过 asserted
  precision、false Skill update 和 dev tokens 三项；F1 gain、coverage、contradiction recall、
  target F1 四项失败，因此 `decision=no_go_dev`。
- Dev cost T0=`120 calls / 102,969 tokens`，T1=`120 calls / 153,466 tokens`，ratio=`1.4904`，
  刚好通过 1.5x 门槛。全 200 cache 的 ratio 超过 1.5，主要包含 selection 的一次三连截断；
  selection 没被打开，所以它不改变 dev gate，但作为真实运行成本保留报告。
- 分动作看，T1 利用历史 pick 参数构造 `at(object,receptacle)` 后，place coverage 从 T0 的
  0.2167 增至 0.3333；但 nav 从 0.0876 降至 0.0584、pick 从 0.1100 降至 0.0050，open/close
  两者均为 0。模型在 dev 只给出 3 个真正 `temporal` tier，却给出 161 个 unknown 和 34 个
  action_prior；额外帧主要增加不确定/重复输出，没有形成可靠的视觉状态记忆。
- 高 asserted precision 与 0 false update 不能视为成功：它来自极低 coverage/大量 unknown，
  正是 Phase3A 已警告的“靠拒绝一切获得安全”。冻结 8B 在本 prompt/预算下不能把三帧 RGB
  当成 structured feedback 的可靠替代，也不能授权 persistent Skill evolution。

### Phase3B 最终决策、停止分支与 artifact（2026-08-25）

- Final decision=`no_go`：P3B-1=`no_go`（唯一失败为 repetition burden）；
  P3B-2=`no_go_dev`（能力/coverage/recall 多项失败）。按预注册 conjunction，未收 episode
  60--79、未执行 fresh temporal audit、未实现/审计 late feedback fusion；三项均标为
  `canceled_by_preregistered_no_go`，不是遗漏。
- 对项目的结论不是“永远依赖 simulator ground truth”，而是：当前 frozen Qwen3-VL-8B 可以用
  短时历史改善 executor 的均值，但仅凭 RGB history 还不足以形成可写长期 Skill 的证据。
  因而近期系统应保留 feedback 作为 episode-control 信号，同时禁止 feedback-only 或
  action-prior-only claim 单独触发 persistent update；不要把本轮 temporal provider 接入 VTCA v2。
- Final artifact：`running/Phase3/phase3b/phase3b_final_decision_20260825.json`，SHA256=
  `98a8d1425513d61599a88c7fe605e1c37418fdd365b7272600eb5697ae968774`。
  Executor analysis SHA=`110e010d68a126081ede1379852125f6fba941711993db0aace55991f65f6a2a`；
  evidence analysis SHA=`a1c4a9701da549930eb470d33517d6642166c422432e40af533f1f0ac85b09d8`；
  T0/T1 cache SHA 分别为 `f5ebce67157f4590df25b1d4f1d41c6af4dcf31251ef1e1d23fa8254b1aea84d`、
  `ba088edfa48b36c5c871a74cafd9fe3e57f22fc3f0f48f2e6bca7b587a6968bc`。
- Finalizer 检查通过：四臂 episode exact pairing、executor event hash、strict cache id/order/hash、
  information isolation 和 audit-read=0 全为 true。新增/修改后仓库完整测试套件 232 条通过。

## 3. Phase 3C：Frozen Meta-Skill-Guided Safe Evolution（预注册于 2026-08-25）

### 3.1 研究边界与主问题

Phase3A/3B 的 No-Go 不能一概解释为 Skill evolution 失败：永久不可观测状态、冻结 8B 的
视觉能力上限、以及所有传感/结果信号同时失效，分别属于环境/硬件、backbone 和可识别性边界。
但普通 partial observability 仍在范围内：若合法移动、重观察或后续图像能够恢复信息，何时触发
这些行为仍是可由 Skill 改善的执行能力；偶发 feedback/visual conflict 也仍要求系统 fail closed，
不能污染 canonical Skill。

Phase3C 因而只检验：

> 在相同冻结 Qwen3-VL-8B、相同环境接口和预算下，当正确 Skill 确实能改善 affected tasks 时，
> 三个短小、环境无关且冻结的 Meta-Skill 能否提高可归因、可修补、最终可接受的有益 Skill
> update yield，而不依赖参数训练或环境专属维护。

`training-free` 在本阶段严格定义为：不更新任何模型权重、不训练额外 verifier、不准备
环境专属监督训练集。它不等于 evaluation-free；candidate promotion 仍必须依赖与 acquisition
隔离的 paired outcome。Meta-Skill 可以由人工/闭源模型在方法设计期一次性起草和审查，但主实验
中不得按 EB-HAB/NAV outcome 反复改写；本阶段实际采用仓库内人工冻结版本，在线 executor、
teacher 和 patch author 均为同一个本地 Qwen3-VL-8B。

### 3.2 P3C-0：Skill-addressable admission

主实验 fault 必须先证明是 Skill-addressable：在完全相同的 8B、观察、feedback、task 和 seed 下，
correct/repaired Skill 相对 faulty Skill 的 affected mean delta > 0、paired bootstrap 95% LCB > 0，
且 protected LCB > -0.05。未通过者不用于否定 Meta-Skill，只作为 model/environment/task-density
boundary 报告；不得在看到 Phase3C Meta-Skill outcome 后修改 admission。

Phase2 已消费结果只作**历史筛查**：`effect_pick_inversion` 曾存在一个 independent-audit
beneficial candidate（affected mean=0.0821、LCB=0.0051、protected LCB=0），暂定为 primary
live fault；`constraint_pick_multihold` 的最佳 affected candidate protected LCB=-0.0856，未通过，
只作离线 patch/stress fault。若进入 live，仍须用 fresh Phase3C paired coordinates 复验 admission，
不能把上述 post-hoc reanalysis 当作新主结果。

### 3.3 P3C-1：三个冻结 Meta-Skill 与访问边界

只维护三个简短 artifact，总文本预算 <=1000 whitespace-delimited tokens：

1. `observe_and_recover`：根据 current action、可用 observation capability 和 unknown evidence 决定
   观察/一次 bounded recovery/暂缓；不得看到 expected delta、Skill predicted polarity、attribution、
   patch 或 outcome label。
2. `attribute_and_scope`：只在独立 evidence packet 完成后读取 mismatches/context，区分
   belief refresh、execution lapse、Skill defect 和 abstain，并只为 Skill defect 定位五字段。
3. `patch_and_test`：只读取已通过 recurrence 的 cited cluster 和 implicated field，生成一个
   exact-target bounded patch、scope 和可证伪 test intent；不得决定 promotion。

实际 gate、schema check、paired bootstrap、protected regression 和 rollback 仍由确定性代码执行。
三个 artifact 禁止出现 `EmbodiedBench`、`EB-HAB`、`EB-NAV`、具体 subset/scene/episode/object
instance；文本、schema 和 combined digest 在任何模型 outcome 前冻结。执行 prompt 每次只披露当前
阶段的 Meta-Skill，不能把三份文本全部无条件塞给 8B。

### 3.4 P3C-2：离线 attribution/patch/隔离 gate

先使用已存在、audit-free 或已明确弱标签性质的数据，避免新 Habitat rollout：

- synthetic fault cases：12 类 x 2 cases，沿用 Phase2 direct-Qwen 五 request seeds；current direct
  teacher 结果只读复用，Meta-Skill teacher 对相同 cases 跑五 seeds；另做 predicate/entity/action
  lexical rename、mismatch order permutation 和 evidence-ID rename 的 deterministic metamorphic test；
- natural weak-gold：`effect_pick_inversion` 与 `constraint_pick_multihold` 的 Phase2 acquisition
  events，只重放 mismatch/context，不发送 RGB；Meta 与 current direct teacher 在相同 case/seed 上
  比较。既有 current artifact 可复用，缺失的 matched seed 才补调用；
- patch：两种 compiled faults 各 10 request seeds，current 结果只读复用；Meta patch 必须同时修复
  target compiled rule、保留无关规则、通过 bounded applier，并输出通用 scope/test intent；
- isolation：单元测试证明 observe payload 不含 Skill/expected/outcome，attribute payload 不含
  images/oracle/audit，patch payload 只含 cited cluster/field；Meta 文本静态 lint 与 digest 必须通过。

离线 Go 同时要求：

1. synthetic Meta direct target Macro-F1 相对 current direct 至少 +0.10，field Macro-F1 不下降超过
   0.02，invalid=0；
2. 两个 natural weak-gold 集合上 Meta 的 Skill-update F1 均不低于 matched current，且合并
   false Skill-update 不增加；
3. rule-first+Meta 不低于 current rule-first 的 target/field 指标；
4. 两类 patch 各 >=9/10 完整通过，且无 instance/benchmark-specific wording；
5. metamorphic target/field agreement >=0.90；
6. 三 Meta-Skill 总文本预算、访问隔离和 frozen digest 全部通过。

任一核心 gate 失败，则取消高成本 live evolution；允许单独继续 P3C-3 的 executor-side
`observe_and_recover` 小 pilot，但其结果只能支持 execution helper，不能支持 Meta-Skill-guided
evolution。

### 3.5 P3C-3：条件性 executor 与受控 fault pilot

若 attribution/patch 离线 Go，先在 30 个固定 paired EB-HAB episodes 比较 current Skill prompt 与
`observe_and_recover`，保持 feedback/history/model/seed/budget 相同。主要指标 task progress；
invalid、adjacent repetition 和 tokens 为安全/成本指标。screening Go：mean progress 严格提高、
不增加 success regression、invalid/repetition 各不超过 `max(1.25*current, current+0.02)`，tokens
<=1.10x。

随后只对 fresh-admitted `effect_pick_inversion`、seed 0 运行 corrupted frozen、
EmbodiSkill*+Common Gate、current VISTA 和 Meta-Skill VISTA；正确 Skill 为 admission/ceiling，
不参与生成证据。所有 arm 匹配 executor/teacher/initial fault/acquisition/candidate/gate/audit budget。
Live Go 必须观察到至少一个 online accepted update，并在 independent audit 上 affected mean > 0、
affected LCB > 0、protected LCB > -0.05；只减少 update 或增加 abstain 不算成功。

### 3.6 P3C-4：条件性 EB-NAV 零样本检查

只有离线 Go 且 executor helper 不退化后才打开。三个 Meta-Skill 的字节与 digest 必须与 EB-HAB
完全相同，只允许替换 action/observation capability adapter。当前 EB-NAV 只支持 frozen evaluate，
因此本阶段先做通用 event/metamorphic replay 和小规模 executor prompt pilot；完整 live evolution
adapter 只有在零样本 pilot 通过后才实现，不因缺少 train-validation split 伪造主结果。

### 3.7 P3C 最终决策

- `go`：离线、live accepted-beneficial 和 protected non-inferiority 均通过，接回 VTCA v2、
  SkillFaultBench 和 release-grade Gate 路线；
- `partial_go_execution_only`：只改善 executor，不改善 reliable evolution；仅保留
  `observe_and_recover` 为辅助/消融，不作为 VISTA 核心演化贡献；
- `no_go`：离线能力或 live beneficial-update yield 不通过，停止 Meta-Skill 扩展；
- 被前置 gate 取消的分支明确记录 `canceled_by_preregistered_no_go`，不记为未完成。

所有新 artifact 写入 `running/Phase3/phase3c/`，existing artifact 只读且禁止覆盖。长任务使用 detached
`tmux`，记录命令、session、PID、endpoint、开始/结束时间、exit code、代码/Meta-Skill/artifact
digest、calls/tokens 和因果调整。

### 3.8 P3C-1 实现冻结、静态检查与运行环境（2026-08-25）

- 在读取任何 Phase3C 新模型 outcome 前冻结三条英文 Meta-Skill，总 whitespace-token proxy=`423`，
  version=`phase3c_frozen_v1`，combined SHA256=
  `ec349ec8bbb0a6dced55267fc0d44c0259a4772f384b3adab765fa0faeff43cc`；快照位于
  `configs/phase3c_meta_skills_frozen_v1.json`。静态 lint 禁止 benchmark 名称和 numbered instance
  identifier，预算小于预注册的 1000。
- 新增 `vista_skill/meta_skills.py`；`JsonAttributionTeacher` 与
  `JsonBoundedPatchGenerator` 只在 Phase3C arm 注入对应 stage 文本，purpose 分别为
  `vista_meta_attribution` / `vista_meta_bounded_patch`；默认路径和 Phase2 artifact 行为保持不变。
  planner 只在 `--meta-skills frozen_v1` 时附加 `observe_and_recover`。CLI 要求 experiment 同时为
  `--diagnostic --method full`，受控默认运行 fail closed。
- stage contract 的含义是防止 oracle/audit/outcome/expected/patch 跨边界泄漏：observe provider 本身是
  无动态输入的 frozen bytes，attribute payload 只有 mismatch/context，patch payload 只有 cited
  cluster/field/compiled contract。需要如实限定：同一个 executor 的完整 prompt 仍正常包含 active
  Skill 和 ledger；prompt section 不能提供模型 attention-level 隔离。本轮未进入 live executor，故没有
  把静态 contract 夸大成已验证的行为隔离。
- 第一次相关测试的 2 个 failure 均来自新测试夹具误用既有 API（漏传 `FixedActionSchema`、把
  `replace_exact` 写成不存在的 `replace`）；在任何 Phase3C 模型调用前只修夹具、不改 Meta-Skill
  文本。之后相关测试 62 条通过，启动前完整 suite=`235 passed`。
- 本机 RTX 4090 上以 detached tmux `phase3c_vllm` 启动同一 frozen
  `Qwen/Qwen3-VL-8B-Instruct`，FP8、TP=1、max model len=16384；项目 endpoint probe 6/6 通过
  （models/plain/seed/schema/trajectory/multimodal）。离线 driver 在 `phase3c_offline` detached tmux
  中运行，SSH 断开不影响；runtime/start/completion manifest 和完整 stdout 分别位于
  `running/Phase3/phase3c/runtime/`。运行过程中没有 endpoint error 或 malformed JSON。

### 3.9 P3C-2 frozen offline gate 结果

所有 current arm 只读复用 Phase2 artifact；Meta arm 完整运行 5 request seeds。未在看到首个负结果后
修改提示、阈值或 seed。离线 finalizer 的 conjunction 结果为：

| 检查 | Current | Frozen Meta-Skill | 预注册要求 | 结果 |
|---|---:|---:|---:|---|
| synthetic target Macro-F1 | 0.5279 | 0.4321（delta=-0.0958） | delta >= +0.10 | Fail |
| synthetic field Macro-F1 | 0.9387 | 0.2963（delta=-0.6424） | delta >= -0.02 | Fail |
| multihold natural Skill-update F1 | 0.5912 | 0.1538 | Meta >= current | Fail |
| effect natural Skill-update F1 | 0.5821 | 0.4286 | Meta >= current | Fail |
| multihold rule-first Skill-update F1 | 0.8571 | 0.8571 | non-inferior | Pass |
| effect rule-first Skill-update F1 | 0.6250 | 0.6250 | non-inferior | Pass |
| compiled patch | current 10/10 + 10/10 | 10/10 + 10/10 | each >=9/10 | Pass |
| environment-neutral patch wording | — | 20/20 | 20/20 | Pass |
| metamorphic target/field agreement | — | 0.9444，invalid=0 | >=0.90 | Pass |
| frozen digest/access/budget | — | 423 tokens，digest match | all pass | Pass |

- synthetic 五个 seed 完全一致：7/7 abstain negatives 保留，但 10 个 true Skill updates 中只有 2 个
  保留，其余 8 个变为 abstain；7 个 belief refresh 中也只有 2 个保留。不是随机 seed 波动，而是稳定的
  conservative bias。
- natural direct Meta teacher 在 multihold 12 个 positives 中只召回 1 个（recall=0.0833），在 effect
  11 个 positives 中召回 3 个（recall=0.2727）。它确实把 weak-gold control 上的 direct false update
  降到 0，但代价是 true-update yield 大幅下降，所以 Skill-update F1 未过。control 并无独立完整标注，
  该 precision 现象只作机制诊断。
- rule-first+Meta 与 recorded rule-first 在两种 fault、全部五 seed 上逐项相同，合并 false Skill update
  均为 0。结论是现有确定性 provenance/rule-first 路由成功隔离了 teacher 的坏偏置，但 Meta-Skill
  没有带来任何额外归因收益。
- 两类 patch 都修复 compiled target rule 且保留同字段无关规则，表面上通过；但 current Phase2 已经
  同样是两类 10/10，且当前 generator 会按 cited evidence 确定性纠正 compiled rule。因此这是
  “未破坏已饱和路径”，不是 Meta-Skill 的增量贡献。8B 还在 20/20 rationale 中都没有完整写出要求的
  replay+compare test intent，作为 instruction-following 缺口保留，但不事后加入 frozen gate。
- metamorphic 72 个变体只有 4 个 disagreement，全部来自 `unknown_as_false` 在 identifier/entity rename
  后由 abstain 变为 belief_refresh；agreement 仍过 0.90。高一致性只说明策略稳定，不能抵消其稳定地
  过度 abstain。

### 3.10 成本、原因分析与 Phase3C 最终 No-Go

- Phase3C 离线共 `676 calls / 424,278 prompt / 131,946 completion / 556,224 total tokens`；无 executor
  call、无 Habitat rollout。matched core（synthetic+natural+patch，不含额外 metamorphic）current=
  `348,393 tokens`，Meta=`498,640 tokens`，Meta/current=`1.4313`。新增 43.1% token 成本却未提升
  attribution 或 patch 成功率。
- 主要失败机制是 `attribute_and_scope` 对当前 8B 形成了过强的 abstention prior。提示中的
  “prefer recurring or independently supported evidence; otherwise abstain”在单 transition 输入上被模型
  稳定地执行成近似硬 veto；它没有引入新 evidence，只是在同一 evidence 上增加文字检查表，因而不能
  消除 causal ambiguity。这个解释同时得到 synthetic 与两个 natural fault 支持。
- `decision=no_go`：offline 五类 gate 中 synthetic=false、natural=false，patch/metamorphic/
  access-and-freeze=true。不可将“错误更少但几乎不更新”写成 reliable Skill evolution improvement，
  也不可用 current rule-first 的保护效果替 Meta-Skill 记功。
- 按 3.4/3.5/3.6 的预注册顺序，fresh Phase3C live fault admission、30-episode EB-HAB executor pilot、
  EB-HAB live four-arm evolution、EB-NAV zero-shot prompt pilot 与 EB-NAV live adapter 全部标记
  `canceled_by_preregistered_no_go`。本轮因此不对 `observe_and_recover` 的 execution-only 效果作正负
  结论，也没有 EB-HAB/NAV live performance claim。
- Phase3C v1 不接入 VISTA 主方法。近期保留现有 rule-first provenance、deterministic compiled-rule
  correction 和 candidate gate；不继续用更长的通用文字 checklist 堆叠同一个 8B。若另开新阶段，
  必须作为新假设重新预注册，例如只研究 executor-side observation Skill，或把正反例/决策表编译为
  可执行的结构化 Meta-Skill；不得修改 v1 prompt 后覆盖本轮 No-Go。
- 决策 artifact=`running/Phase3/phase3c/offline/phase3c_offline_decision.json`，SHA256=
  `c0d580e28639ffff29470b6a9b579769487993455a20cdc1617461d3a04c80d8`；final analysis=
  `running/Phase3/phase3c/phase3c_final_analysis_20260825.json`，SHA256=
  `762080671728e2d8a43a505a325c21a42fe4e96d15a2d9a2ebc42dd7f21352e2`。synthetic/meta-metamorphic
  SHA256 分别为
  `9b1ad4fc8b678d4641bd01a21ec520e0e1d93b49318c1942555f5d200a0b4f92`、
  `1c578c39fb0b657689ee088d4489ea133b7ededfb9aae342dc6365a32a0ce430`。
- 离线 tmux 正常输出 `PHASE3C_OFFLINE_COMPLETE` 后退出；只为本轮启动的 `phase3c_vllm` 随后关闭，
  GPU 回到 `106 MiB / 0%`。未删除任何 artifact，服务可由原脚本恢复。
