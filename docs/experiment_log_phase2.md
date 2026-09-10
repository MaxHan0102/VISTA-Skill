# VISTA-Skill Phase 2 Experiment Log

> 2026-09-08 目录整理：本阶段实验实体已归档至 `running/Phase2/`，路径引用已改写且兼容软链接已删除，原版已备份。引用的 Phase1 源数据仍归 Phase1，详见[实验输出目录说明](experiment_output_layout.md)。

本文件记录 2026-08-24 开始的 Phase 2 验证。所有结果均使用
`Qwen/Qwen3-VL-8B-Instruct`，历史 `running/` artifact 未被覆盖。

## 环境与端点

- Habitat client：本机 RTX 4090 D，单进程串行运行，避免 EGL/显存竞争。
- Qwen endpoint A：`http://192.168.1.185:8000/v1`。
- Qwen endpoint B：`http://192.168.1.173:8001/v1`。
- 两个端点均报告 `max_model_len=16384`，此前 serving contract 6/6、method
  smoke 3/3 均通过。
- A 用于完整 Habitat campaign；B 用于不占模拟器的 teacher、patch 和
  natural-evidence 诊断。

## P2-0：E8p Gate 重分析与 compiled-rule 缺陷

### 问题与修复

E8p 的 `_evidence_corrected_rules()` 使用 predicate 名称决定需要修改的
compiled rule。`constraint_pick_occupies_gripper` 和
`constraint_place_frees_gripper` 都预测 `not_holding`，所以一次 pick
contradiction 会把两条规则同时改为 `false`：修复 pick 的同时破坏 place。

实现现已改为按 `ExpectedChange.source_id` 的完整 provenance 精确匹配规则；
回归测试要求 pick=`false` 且未被归因的 place=`true`。相关 53 个 targeted
tests 和完整 pytest suite 均通过。

### E8p 离线重分析

命令：

```bash
PYTHONPATH=. /root/miniconda3/envs/max_embench/bin/python \
  scripts/phase2_gate_reanalysis.py \
  --output running/Phase2/phase2_gate_reanalysis_20260824.json
```

affected group 在查看 outcome 前按 fault semantics 定义为
`goal_preds.inputs` 至少包含两个 object variables 的任务，其余为 protected；
bootstrap 10,000 次，alpha=0.05，protected non-inferiority margin=-0.05。

结果：

- E8p 7/7 候选均发生 unrelated place-rule corruption。
- 7/7 在独立 audit 中为 harmful。
- 0/7 通过 post-hoc affected-LCB>0 且 protected-LCB>-0.05 的诊断策略。
- 候选 `59975891618b...` 虽然 global delta `+0.0174`，affected delta
  `-0.1252`；正 global mean 不能说明修复有效。
- 候选 `def6e2e2a8a2...` affected delta `+0.2054`，但 protected delta
  `-0.0426` 且其 confidence bound 未通过；同样不能据此接受。

结论：旧 E8p 不能用于证明 subgroup gate 应放宽。它首先证明候选实现发生了
同 predicate 的规则串改；threshold 仍必须在新的、实现正确且 split 独立的数据上
预注册验证。

## P2-1a：Oracle/Noisy Evidence 敏感性

命令：

```bash
PYTHONPATH=. /root/miniconda3/envs/max_embench/bin/python \
  scripts/phase2_evidence_noise_sweep.py \
  --output running/Phase2/phase2_evidence_noise_20260824.json
```

设置：300 synthetic oracle requests，20 noise replicates，12 fault families x
10 cases；分别扫描 drop、polarity flip、false positive 和三者组合。

关键结果：

| Evidence noise | Predicate F1 | VTCA target F1 | VTCA field F1 | ECE |
|---|---:|---:|---:|---:|
| 0 | 1.000 | 1.000 | 1.000 | 0.000 |
| drop=flip=FP=0.05 | 0.907 | 0.906 | 0.930 | 0.071 |
| drop=flip=FP=0.10 | 0.825 | 0.813 | 0.842 | 0.133 |
| drop=flip=FP=0.20 | 0.661 | 0.668 | 0.695 | 0.257 |
| drop=flip=FP=0.30 | 0.519 | 0.530 | 0.541 | 0.362 |

单一 20% polarity flip 已使 predicate F1 降至 `0.803`；30% drop 使 recall
降至 `0.701`。该实验验证 evidence error 会直接传导为 attribution error，但
synthetic noise 不是自然 Habitat accuracy 的估计。

## P2-1b：自然 transition 的 Qwen 弱标签 Evidence Audit

命令：

```bash
PYTHONPATH=. OPENAI_API_KEY=EMPTY \
  /root/miniconda3/envs/max_embench/bin/python \
  scripts/phase2_natural_evidence_weak_audit.py \
  --events running/Phase1/fault_repair_e8p_constraint/full/seed_0/acquisition.jsonl \
  --base-url http://192.168.1.173:8001/v1 \
  --output running/Phase2/phase2_natural_evidence_weak_audit_173_20260824.json
```

设置：45 个真实 Habitat transitions，50 个 simulator-feedback definite
predicate targets；每个 event 分别运行 public-feedback+images 和 images-only，
共 90 次 Qwen 多模态调用，无 API failure。

| 条件 | Coverage | Precision | Recall | F1 | ECE | Brier |
|---|---:|---:|---:|---:|---:|---:|
| public feedback + images | 0.860 | 0.953 | 0.820 | 0.882 | 0.046 | 0.047 |
| images only | 0.760 | 0.632 | 0.480 | 0.545 | 0.341 | 0.341 |

images-only 分项：pick F1=`0.440`，nav=`0.750`，place=`1.000`，open=`0`，
close=`0`。其中 pick 有 27 targets，其他 action 样本很少。images-only 的
confidence>=0.8 selective risk 仍为 `0.343`，pick 为 `0.524`。

结论：当前 evidence branch 的较高总体结果主要依赖 public simulator feedback；
纯视觉尤其不能可靠判断 pick 后的 `holding/not_holding`。该结果使用 feedback
predicate 作为 weak gold，feedback-enabled arm 还把同一 feedback 文本暴露给模型，
因此不能替代原计划的 300-event 人工/状态 gold audit。

### P2-1c：三臂 Evidence 信息源消融

为区分视觉输入相对 public feedback 的增量价值，在相同 45 transitions / 50
targets 上补充 feedback-only arm；三臂重新独立调用 Qwen，共 135 calls。Artifact：
`running/Phase2/phase2_natural_evidence_three_arm_173_20260824.json`。

| 条件 | Coverage | Precision | Recall | F1 | Prompt + completion tokens |
|---|---:|---:|---:|---:|---:|
| images + feedback | 0.880 | 0.955 | 0.840 | 0.894 | 37,040 |
| feedback only | 0.700 | 1.000 | 0.700 | 0.824 | 10,042 |
| images only | 0.760 | 0.632 | 0.480 | 0.545 | 41,378 |

images + feedback 相对 feedback-only 只增加 `+0.070 F1`，同时本次调用 token
约为其 3.69 倍；images-only 仍显著较弱。这个消融进一步说明当前 evidence
质量和效率主要由公开反馈支撑，而不是纯视觉转移识别。由于 weak gold 本身由
feedback 派生，feedback-only 会天然占优；该结果只能量化当前 weak-audit
分布上的增量，不能作为独立视觉准确率结论。

### P2-1d：Evidence request-seed 与跨 campaign 复现

三臂脚本新增显式 `--request-seed`，默认值仍为 0。旧 E8p acquisition 上的
两个额外有效 seeds 均得到 images+feedback / feedback-only / images-only F1 =
`0.882 / 0.824 / 0.545`；原 seed 0 为 `0.894 / 0.824 / 0.545`。融合视觉相对
feedback-only 的增量稳定在 `+0.058` 至 `+0.070`。

在 provenance 修复后的 multihold acquisition 上独立重放 47 transitions / 53
targets，三臂 F1 为 `0.866 / 0.795 / 0.500`，视觉增量 `+0.071`。绝对值略降，
但信息源排序和差距跨 campaign 一致。Artifacts：

- `running/Phase2/phase2_natural_evidence_three_arm_seed1_retry_173_20260824.json`；
- `running/Phase2/phase2_natural_evidence_three_arm_seed2_173_20260824.json`；
- `running/Phase2/phase2_multihold_natural_evidence_three_arm_seed0_173_20260824.json`。

另有一个首次 seed-1 artifact 的 135 calls 全部为本地网络沙箱
`APIConnectionError`，其零 coverage 不纳入任何指标；该失败 artifact 被保留为
运行审计记录。

## P2-2：Qwen Teacher 与 Patch 稳定性

### Teacher fallback smoke

在 endpoint B 上真实触发一次 attribution fallback 和一次 bounded patch：3/3
通过；usage 为 attribution 378 prompt + 112 completion tokens，patch 272 + 177。

### Teacher-only attribution 对照

命令：

```bash
PYTHONPATH=. OPENAI_API_KEY=EMPTY \
  /root/miniconda3/envs/max_embench/bin/python \
  scripts/phase2_teacher_attribution.py \
  --base-url http://192.168.1.173:8001/v1 --seeds 5 \
  --output running/Phase2/phase2_teacher_attribution_173_20260824.json
```

在同一个 24-case synthetic fault set 上跳过 rule-first，直接调用 Qwen teacher。
5 个 request seeds 的输出完全一致：target Macro-F1=`0.528`，field
Macro-F1=`0.939`。Qwen 把 7/7 belief-refresh cases 全部判为 abstain，并漏掉
1/10 skill-update；120 calls 无 invalid response，合计 28,975 prompt + 15,748
completion tokens。作为对照，rule-first VTCA 为
target/field F1=`1.0/1.0`，trajectory always-skill-update target F1=`0.196`。

结论：在该 taxonomy-aligned diagnostic 上，主要增益来自 provenance-aware
algorithmic routing，不是 Qwen3-VL-8B teacher 本身；teacher 更保守，尤其不能区分
belief refresh 与 abstention。仍需自然事件 gold 才能外推到真实 attribution。

### 自然 fault-injected transition 归因审计

命令：

```bash
PYTHONPATH=. OPENAI_API_KEY=EMPTY \
  /root/miniconda3/envs/max_embench/bin/python \
  scripts/phase2_natural_attribution_audit.py \
  --events running/Phase2/phase2_multihold_provenance_fix/full/seed_0/acquisition.jsonl \
  --base-url http://192.168.1.173:8001/v1 \
  --output running/Phase2/phase2_natural_attribution_weak_audit_173_20260824.json
```

从自然 Habitat acquisition 中预先按 injected fault provenance 定义正例：
`constraint_pick_occupies_gripper` 的 covered contradiction 为
`skill_update(constraint)`；其余 feedback-observable Skill mismatch 作为 control。
共 42 events，12 positive；这是 counterfactual weak gold，不把 control 中可能存在的
其他自然 Skill defect 当作已完成独立标注。

| 方法 | Target Macro-F1 | Field Macro-F1 | Update precision | Update recall |
|---|---:|---:|---:|---:|
| recorded rule-first VTCA | 0.905 | 0.905 | 1.000 | 0.750 |
| fault-relevant oracle filter | 1.000 | 1.000 | 1.000 | 1.000 |
| direct Qwen teacher, seed 0 | 0.595 | 0.401 | 0.414 | 1.000 |
| direct Qwen teacher, seed 1 | 0.618 | 0.409 | 0.429 | 1.000 |

Rule-first 的 3 个 false negatives 都是同一结构：一个充分支持的 fault
contradiction 与无关的 termination unknown/uncovered mismatch 同时出现；当前
`CreditAssigner` 的全量 insufficient-evidence 前置检查使整个 transition abstain。
oracle filter 只用于定位这项混合 mismatch 瓶颈，不是可部署方法。直接 teacher
则把 30 个 controls 中的 16-17 个错误路由为 Skill update。两 request seeds 结论
稳定：不能简单用 teacher 替换规则；下一版应研究 provenance-aware mismatch
partition / dominance，同时保留 control false-positive veto。

补充到 5 个 request seeds 后，direct Qwen 在当前 campaign 的 30 个 controls 中
产生 16--17 个 false positives；target Macro-F1 仅 `0.595--0.618`，field
Macro-F1 仅 `0.401--0.409`。在旧 E8p campaign 的 39 cases 上复现时，rule-first
precision=`1.0`、recall=`0.667`，而 direct Qwen precision=`0.321`、19/30 controls
被错误更新。结果跨 seed、跨 acquisition 一致：teacher 随机性不是主要瓶颈。

### Provenance partition 后验机制诊断

新增一个仅存在于审计脚本、尚未接入核心方法的诊断臂：当且仅当高置信、
Skill-sourced decisive contradiction/missing-progress 指向唯一字段，且不存在
identity、stochastic、action-schema 或 termination 冲突时，允许该 partition
压过无关的 unknown/uncovered mismatch。首次版本在当前 42 cases 上将 recall 从
`0.75` 提到 `1.0`，但对一次未执行的 invalid pick 产生错误 update。检查该失败后，
规则增加 execution-lapse 约束：`executor_followed_skill=false` 时，
missing-progress 不可压过旁路证据，只有 contradiction 可以。

修订规则在旧 E8p campaign（未用于上述修订）上达到 target/field Macro-F1
`1.0/1.0`，9/9 positives 全召回、30/30 controls 零更新；在用于分析的当前
campaign 上后验结果也是 `1.0/1.0`。这支持把 partition/dominance 作为下一版
预注册算法假设，但当前 gold 只标记 injected fault，且规则经过一次 post-hoc
修订，不能当作独立方法主结果。Artifacts：

- `running/Phase2/phase2_natural_attribution_partition_audit_seed0_173_20260824.json`；
- `running/Phase2/phase2_e8p_natural_attribution_partition_audit_seed1_173_20260824.json`；
- `running/Phase2/phase2_natural_attribution_partition_refined_audit_seed1_173_20260824.json`。

### 自然 episode-level Trajectory Reflection 对照

命令：

```bash
PYTHONPATH=. OPENAI_API_KEY=EMPTY \
  /root/miniconda3/envs/max_embench/bin/python \
  scripts/phase2_natural_trajectory_attribution.py \
  --events running/Phase2/phase2_multihold_provenance_fix/full/seed_0/acquisition.jsonl \
  --base-url http://192.168.1.173:8001/v1 --seeds 2 \
  --output running/Phase2/phase2_natural_trajectory_attribution_173_20260824.json
```

把同一 20 个自然 episodes 压成 EmbodiSkill-style trajectory summary；只要 episode
包含预定义 multihold contradiction 就标为 update positive，共 11 positives。两个
request seeds 输出完全一致：target Macro-F1=`0.495`，update precision/recall/F1
均为 `0.545`，field Macro-F1=`0.111`。所有 persistent proposals 都归到
`procedure` 或 `activation`，`0/11` 命中真正的 `constraint` field。

这项结果比 synthetic always-update 对照更接近真实 baseline 输入：trajectory
reflection 能看到部分失败模式，但缺少 predicate provenance，无法定位 compiled
multihold rule。它预示 EmbodiSkill*+Common Gate 即使生成候选也难以修复目标 fault；
仍需端到端 campaign 验证 gate 和最终 frozen Skill 行为。

补充到 5 个 request seeds 后，五次输出逐 episode 完全一致，所有指标不变，且
仍为 `0/11` 命中真正的 `constraint` 字段。Artifact：
`running/Phase2/phase2_natural_trajectory_attribution_5seeds_173_20260824.json`。因此该
失败不是 decoding seed 偶然性；端到端 Common Gate baseline 的前置风险明确位于
trajectory proposal attribution，gate 本身不能恢复错误字段。

### Multihold patch stability

命令：

```bash
PYTHONPATH=. OPENAI_API_KEY=EMPTY \
  /root/miniconda3/envs/max_embench/bin/python \
  scripts/phase2_patch_stability.py \
  --base-url http://192.168.1.173:8001/v1 --trials 10 \
  --output running/Phase2/phase2_patch_stability_173_20260824.json
```

结果：10/10 seeds 通过；10 次均产生同一个 patch/candidate/text，bounded apply
全部成功，compiled rule 始终为 pick=`false`、place=`true`。这证明 provenance
修复在 live Qwen patch path 上稳定，但还不证明 rollout benefit。

同一脚本的 `--fault effect_pick_inversion` 前置筛查也是 10/10 seeds 通过；
10 次均产生文本 `Picking up an object results in holding the object.`，目标
`effect_pick_holds_target_category=true`，其他 effect compiled rules 保持不变。
Artifact：`running/Phase2/phase2_patch_stability_effect_173_20260824.json`。

## P2-3：修复后的 Habitat Fault-Repair Campaign

状态：multihold Full 与 EmbodiSkill*+Common Gate baseline 已完成；第二 fault
family `effect_pick_inversion` 正在运行。

命令：

```bash
PYTHONPATH=EmbodiedBench:. CUDA_VISIBLE_DEVICES=0 OPENAI_API_KEY=EMPTY \
  /root/miniconda3/envs/max_embench/bin/python \
  -m vista_skill.integrations.embodiedbench.cli experiment \
  --method full --config configs/vista_fault_repair_fullsel_p10.json \
  --manifest configs/eb_hab_train_validation_manifest.json --diagnostic \
  --max-acquisition-episodes 20 --evolution-seeds 0 \
  --executor-base-url http://192.168.1.185:8000/v1 \
  --method-model Qwen/Qwen3-VL-8B-Instruct \
  --method-base-url http://192.168.1.185:8000/v1 \
  --skill-fault constraint_pick_multihold \
  --output-dir running/Phase2/phase2_multihold_provenance_fix/full
```

### Multihold Full VISTA 最终结果

Acquisition 20 episodes：success=`14/20`，mean progress=`0.7625`，171
transitions。归因计数为 abstain=`143`、skill_update=`18`、belief_refresh=`10`。
共 15 次 proposal attempts：12 个 materialized candidates、8 个 unique hashes、
4 次重复候选、3 次 static non-materialized failures；0 accepted。拒绝原因：9 次
paired gate、3 次 transition consistency、3 次 static。

12 个 candidates 均在 20 个独立 audit tasks x 3 rollout seeds 上评估，合计 720
candidate pairs；8 个 unique candidates 在 epsilon=0 下全部为 harmful：

| Candidate | Field | Global delta | Affected delta | Protected delta |
|---|---|---:|---:|---:|
| `48776ddc80ac` | constraint | -0.0164 | +0.0166 | -0.0222 |
| `580733761480` | constraint | -0.0450 | -0.0518 | -0.0438 |
| `39e37ba31d09` | termination | -0.0248 | +0.0775 | -0.0428 |
| `1cfd1e5cae81` | constraint | +0.0174 | -0.1252 | +0.0425 |
| `e075425df387` | constraint | -0.0054 | +0.2054 | -0.0426 |
| `e818019fea3a` | constraint | +0.0054 | +0.0772 | -0.0073 |
| `aaf23bb368bc` | termination | -0.0591 | -0.1387 | -0.0451 |
| `b0a1aca419ba` | termination | -0.0227 | -0.0215 | -0.0229 |

0/12 proposal instances 通过 post-hoc affected-LCB>0 且 protected-LCB>-0.05
诊断策略。典型 selection-to-audit reversal 包括：`580733...` proxy
mean=`+0.1020`、finalist=`+0.0368`，但 audit global=`-0.0450`；`1cfd...`
proxy=`+0.2970`，finalist=`-0.0016`，audit affected=`-0.1252`。即使
`e075...` 的 affected mean=`+0.2054` 且 LCB=`+0.0331`，protected LCB
=`-0.0856`，仍不能视为安全 repair。

provenance 修复后 12/12 patches 均未发生 unrelated place-rule corruption；这项
实现缺陷已消除。但最终 frozen Skill digest 仍为
`f3bd8ab91c0d504a26484794177da0ff1a71fe95a3a9b844bfa47018ac08129e`，即原始
故障 v0。严格解释是：gate 成功阻止了已审计的有害候选 promotion，但本 campaign
没有完成 repair 或 Skill evolution，不能用“0 harmful promoted”单独声称有效。

成本：executor 76 calls，354,448 prompt + 27,291 completion tokens；method
204 calls，191,110 prompt + 82,169 completion tokens。其中 visual evidence
167 calls，占 method completion tokens 的绝大多数。正式重分析使用校正后的
artifact `running/Phase2/phase2_multihold_provenance_fix_reanalysis_corrected_v2_20260824.json`。
早期未校正版本把未包含 place rule 的 termination patch 误算为 corruption，已保留
但不得用于结论。

### EmbodiSkill* + Common Gate 端到端 baseline

相同 config/manifest/seed、20 acquisition episodes：success=`13/20`，mean
progress=`0.6500`，mean steps=`8.55`（Full 也为 `8.55`）。trajectory teacher 对
7 个失败 episodes 调用 reflection，共 3,110 prompt + 5,185 completion tokens；
但 `ready_clusters_by_episode` 全为 0，因而 0 proposal、0 candidate、0 gate
rollout，最终 frozen Skill 与 Full 相同，仍是故障 v0。

这与自然 trajectory 5-seed 诊断的 `0/11 constraint` field 命中一致：该 baseline
的失败发生在上游 episode-level attribution/recurrence，而不是 Common Gate 拒绝。
它也没有实现 Skill evolution。executor 成本为 81 calls、371,681 prompt +
30,212 completion tokens。Artifact 目录：
`running/Phase2/phase2_multihold_provenance_fix/embodiskill_star_common_gate/`。

## P2-4：Effect Pick Inversion 自然事件诊断

第二个 fault family 使用 `effect_pick_inversion`：初始 Skill 错误预测 pick 后
`holding(target)=false`。Full acquisition 的 20 episodes 已完成，随后在不启动第二个
Habitat 进程的前提下，使用 `192.168.1.173:8001` 上相同冻结
Qwen3-VL-8B-Instruct 对已记录事件做 teacher-only 诊断。

### Evidence 三条件、三 request seeds

54 个含 public-feedback weak gold 且图像完整的 transitions，共 62 个 predicate
targets。三个条件均为 0 API failures：

| Request seed | Images + feedback F1 | Feedback-only F1 | Images-only F1 | 视觉增量 |
|---:|---:|---:|---:|---:|
| 0 | 0.852 | 0.841 | 0.505 | +0.011 |
| 1 | 0.883 | 0.841 | 0.505 | +0.042 |
| 2 | 0.873 | 0.841 | 0.505 | +0.032 |

融合条件仍优于 feedback-only，但本 campaign 的增量 `+0.011--+0.042` 小于旧 E8p
和 multihold acquisition 的 `+0.058--+0.071`；images-only 仍弱。这说明视觉层
不是完全无效，但目前多数 weak-gold facts 已被 public feedback 覆盖，且视觉自身
不足以支撑可靠 credit assignment。Artifacts：
`running/Phase2/phase2_effect_natural_evidence_weak_audit_seed{0,1,2}_173_20260824.json`。

### Transition-level attribution、五 request seeds

按注入前确定的 provenance weak gold，46 个 feedback-observable Skill-mismatch
events 中有 11 个 `effect_pick_holds_target_category` covered contradictions，应路由
到 `skill_update(effect)`；其余 35 个为 controls。

| 方法 | Target Macro-F1 | Field Macro-F1 | Update precision | Update recall |
|---|---:|---:|---:|---:|
| recorded rule-first | 0.524 | 0.773 | 1.000 | 0.455 |
| post-hoc provenance partition | 0.633 | 0.936 | 1.000 | 0.818 |
| fault-relevant oracle filter | 1.000 | 1.000 | 1.000 | 1.000 |
| direct Qwen, seed 0 | 0.661 | 0.359 | 0.423 | 1.000 |
| direct Qwen, seeds 1--4 | 0.641 | 0.348 | 0.407 | 1.000 |

direct Qwen 在五个 seeds 中均召回 11/11 positives，但把 35 个 controls 中的
15--16 个错误更新。recorded rule-first 零 false positives，但只召回 5/11；
partition 召回 9/11 且继续保持零 false positives，剩余 2 个 positives 被
identity/belief veto 路由成 belief refresh。与 multihold 的 post-hoc 结果相比，
这暴露了下一版 dominance 规则不能简单压过所有 identity signals，需区分真实
instance ambiguity 和与 decisive Skill contradiction 同时出现的旁路 identity
evidence。partition 和 oracle 均为机制诊断，不是独立方法主结果。Artifacts：
`running/Phase2/phase2_effect_natural_attribution_audit_seed{0,1,2,3,4}_173_20260824.json`。

### Episode-level trajectory reflection、五 request seeds

20 个 episodes 中 10 个包含上述 fault contradiction。五次输出逐 episode 完全
一致：target Macro-F1=`0.549`，update precision/recall=`0.556/0.500`，field
Macro-F1=`0.190`。所有预测为持久更新的 positives 都落到 `procedure`，没有一次
命中真实 `effect` field。Artifact：
`running/Phase2/phase2_effect_natural_trajectory_attribution_5seeds_173_20260824.json`。

这与 multihold trajectory 的 `0/11 constraint` 命中形成跨 fault 复现：
episode-level baseline 能识别部分失败需要经验更新，却缺少 transition predicate
provenance 来定位 compiled Skill field。因此其主要瓶颈发生在 Common Gate 之前，
不能假定共享 VISTA gate 就能补救错误候选字段。

### 2026-08-24 中断与 audit 恢复

VS Code/终端关闭时，独立 audit 已有 181 个坐标写出完整 `episode_result`，另有一个
JSONL 只写到中间 event。新增 `scripts/resume_phase2_update_audit.py`：校验 config、
manifest、split 和 snapshot digests，完整坐标作为跨进程磁盘 cache；不完整 JSONL
原样保留，重跑写入 `.resumeN.jsonl`。恢复不会重跑 acquisition、proposal 或已完成
rollout，也不改变任务/seed 配对。

科学指标不受此次中断影响；损失是部分 wall time、少量重复调用，以及原进程仅存于
内存、尚未写入 run manifest 的精确 token usage counters。恢复后的 manifest 必须
把这部分 cost 标为 unavailable，不能从 episode 数伪造 token 数。Full independent
audit 最终已通过 audit-only 恢复完成；下节给出正式结果。

## P2-5：Effect Pick Inversion Full VISTA 最终结果

Acquisition 20 episodes：success=`14/20`，mean progress=`0.725`，mean steps=`8.8`，
176 transitions。归因计数为 abstain=`148`、belief refresh=`20`、skill update=`8`。
共 7 次 proposal attempts，全部 materialized，5 个 unique hashes；0 accepted。拒绝
原因为 paired gate 5 次、transition consistency 2 次。

7 个 proposal snapshots 均在 20 个 audit tasks x 3 rollout seeds 上评估，共 420
candidate pairs；同一个 candidate 的 3 个 proposal snapshots 保留为独立 update
attempts，但复用同一 content-addressed rollout。结果为：

| Candidate | Field | Audit class | Global delta | Affected delta | Protected delta |
|---|---|---|---:|---:|---:|
| `877fa9621cd1` x3 | effect | beneficial | +0.0416 | +0.0587 | +0.0019 |
| `3676fbb4d356` | effect | beneficial | +0.0586 | +0.0821 | +0.0037 |
| `d5504405fb2b` | effect | harmful | +0.0149 | +0.0205 | +0.0019 |
| `99c20ee6b002` | termination | harmful | -0.0218 | -0.0319 | +0.0019 |
| `5175c7d58af3` | termination | harmful | -0.0166 | -0.0245 | +0.0019 |

`877fa...` 和 `3676...` 都把 compiled rule 正确修成
`effect_pick_holds_target_category=true`；前者三次独立 proposal 均产生相同内容。
`d550...` 虽然 compiled rule 也为 true 且 global mean 为正，但两个原 protocol
subgroups 轻微回退，因此按 epsilon=0 被 audit 判 harmful；这再次说明只看全局均值
会掩盖 subgroup regression。两个 termination candidates 均没有修复目标 effect，
且独立 audit 为负。

原始 gate 0 promotion，导致 update audit 的
`missed_beneficial_update_rate=4/7=0.571`。这不是“gate 安全地阻止所有有害 patch”
的单边结果：gate 同时拒绝了 4 个 independently beneficial update attempts，最终
frozen Skill 仍为故障 v0，digest
`3d5e3c1fe8a6d690d8e0a468d52bc2d6c431d65852557234256b304f5a7afd07`。

fault-specific 后验诊断在读 outcomes 前把递归 goal expression 含 `on_top(...)` 的
任务定义为 affected（本 audit 14 tasks），冰箱/纯导航等 6 tasks 定义为 protected。
`3676...` 是唯一通过 affected-LCB>0 + protected-LCB>-0.05 的 proposal：affected
mean=`+0.0821`、LCB=`+0.0051`；protected mean=`+0.0037`、LCB=`0`。然而原 selection
proxy mean=`+0.0946`、LCB=`-0.0158`，因此被拒绝。`877fa...` 的 affected mean
同样为正，但 LCB=`-0.059`，说明不是所有同义 repair 都足够稳定。

这组结果首次在当前项目中真正观察到“可产生、可独立验证但未 promotion”的有益
Skill repair。它支持算法和 benchmark/gate 任务密度共同构成瓶颈：Qwen3-VL-8B
已经能提出正确修复，通用 selection tasks 与 10-task proxy 的统计功效却不足；按
fault-relevant affected/protected 设计可恢复一个稳定有益 candidate，但该策略仍是
post-hoc hypothesis，必须在新 fault、seed 和预注册阈值上独立验证。

Artifacts：

- `running/Phase2/phase2_effect_pick_inversion/full/seed_0/update_audit.json`；
- `running/Phase2/phase2_effect_pick_inversion_gate_reanalysis_20260824.json`；
- `running/Phase2/phase2_effect_pick_inversion/full/seed_0/run_manifest.json`；
- `running/Phase2/phase2_effect_pick_inversion/full/experiment_manifest.json`。

由于原进程在 manifest 写出前中断，acquisition 的 executor/method token counters
不可恢复，manifests 已明确记录为 unavailable；不得据此与 multihold 做精确成本比。

## P2-6：Effect EmbodiSkill* + Common Gate 最终结果

与 Full 使用相同 config、manifest、seed、fault、20-episode acquisition budget 和冻结
Qwen3-VL-8B-Instruct；executor 为 `192.168.1.185:8000`，trajectory teacher 为
`192.168.1.173:8001`，两端权重相同。

Acquisition success=`12/20`，mean progress=`0.625`，mean steps=`8.6`；作为参考，
Full 是 `14/20`、`0.725`、`8.8`，但这里只是同 seed 的描述性比较，不是多 seed
显著性结果。executor 成本为 84 calls、387,039 prompt + 30,190 completion tokens。

trajectory teacher 对 8 个失败 episodes 调用 reflection，共 3,465 prompt + 17,146
completion tokens；但 `ready_clusters_by_episode` 20 项全为 0，因而 0 proposal、
0 candidate、0 gate rollout。`update_audit.json` 为合法空报告，最终 frozen Skill 仍是
故障 v0，digest 与 Full 相同为
`3d5e3c1fe8a6d690d8e0a468d52bc2d6c431d65852557234256b304f5a7afd07`。

该结果与 effect trajectory 5-seed 诊断的 `0` 次 effect-field 命中、以及 multihold
Common Gate 的 0 proposal 跨 fault 一致。严格结论是：EmbodiSkill-style
episode-level reflection 在当前 structured fault 下无法把失败聚合到正确 compiled
Skill field，Common Gate 根本没有机会评价候选；它没有实现 Skill evolution。
Artifact 目录：
`running/Phase2/phase2_effect_pick_inversion/embodiskill_star_common_gate/`。

## P2-7：当前验证覆盖与阶段性结论

按本阶段原定的 8 类验证内容，目前为 **5 项完成、2 项部分完成、1 项未完成**：

| 验证内容 | 状态 | 简要结果与说明 |
|---|---|---|
| 视觉证据三条件消融 | 完成 | Images + feedback 最好，但视觉增益有限；images-only 较弱，视觉层有效但不是充分条件。 |
| Transition-level VTCA 归因 | 完成 | Rule-first 精度高、召回低；直接 Qwen 召回高但误更新多，provenance-aware partition 更值得继续验证。 |
| Qwen patch 生成稳定性 | 完成 | 两类 fault 都能稳定生成正确 compiled-rule repair，8B teacher 的 patch 生成能力不是主要瓶颈。 |
| 两类 fault 的 Full 端到端验证 | 完成 | Multihold 未找到可靠有益 repair；effect 找到 4/7 beneficial attempts，但均未 promotion。 |
| EmbodiSkill* + Common Gate 对照 | 完成 | 两类 fault 都是 0 proposal；失败发生在 trajectory attribution/recurrence，gate 没有机会评估候选。 |
| Gate 与 benchmark 任务密度 | 部分完成 | Effect repair 在相关 pick tasks 上有稳定收益，通用 selection pool 会稀释信号；当前分组仍是 post-hoc，需预注册复现。 |
| 多 seed 可靠性 | 部分完成 | Evidence、attribution、trajectory 已做多 request seeds；高成本 Habitat 端到端 campaign 仍主要是 evolution seed 0。 |
| Teacher 参数规模对照 | 未完成 | 当前只有 Qwen3-VL-8B；尚不能直接量化更大 teacher 是否改善 attribution、candidate diversity 或 gate outcome。 |

阶段性判断：当前主要瓶颈不是 Qwen3-VL-8B 无法生成正确 Skill，而是混合 mismatch
下的归因召回、trajectory-level 字段定位、gate 的统计功效，以及 benchmark 中
fault-relevant tasks 的密度。当前实验已经观察到 independently beneficial Skill
repair，但尚未观察到该 repair 被在线 gate 接受并形成 frozen Skill version 升级，
因此仍不能声称完成真正意义上的端到端 Skill evolution。
