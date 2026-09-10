# VISTA-Skill 实验输出目录

2026-09-08 将已有方法实验的实体文件归档到 `running/Phase1` 至
`running/Phase5`。阶段归属依据各阶段实验日志、实验 manifest 中的
`experiment_id` 和轨迹中的图片引用；不能仅按最近一次被引用的阶段分类。

| 阶段 | 主要内容 | 实体目录 |
|---|---|---|
| Phase1 | 早期 pilot、fault repair、E8 系列、T4、No Skill / Static Skill 导航评测 | `running/Phase1/` |
| Phase2 | 证据噪声、自然归因复查、multihold、effect inversion | `running/Phase2/` |
| Phase3 | Phase3a 证据与 oracle 检查、Phase3b、Phase3c | `running/Phase3/` |
| Phase4 | 人工 Target-Skill 诊断及冻结验证 | `running/Phase4/` |
| Phase5 | discovery、constraint、temporal、P5.6、P5.7 recovery 与 UNKNOWN recheck | `running/Phase5/` |

主要实验保留原文件／目录名，例如：

```text
running/
  Phase1/pilot/
  Phase1/fault_repair_e8p_constraint/
  Phase1/vista_skill/nav_official/
  Phase2/phase2_effect_pick_inversion/
  Phase3/phase3a/
  Phase3/phase3b/
  Phase3/phase3c/
  Phase4/target_skill_oracle_v1/
  Phase5/phase5_p57_recovery_20260907/
  Phase5/phase5_p57_recovery_unknown_recheck_20260907/
```

散落于模拟器输出目录的配套图片归入各阶段的 `simulator/`，其下保留旧路径
层次，以区分仓库根目录运行与从 `EmbodiedBench/` 启动的运行。例如：
`running/Phase5/simulator/EmbodiedBench/running/eb_habitat/`。

历史源数据只存一份：Phase2 对 Phase1 的复查、Phase4 对 Phase1 的轨迹分析，
不会把源轨迹重新归属到后续阶段。12 个未找到直接轨迹引用的图片目录根据
VISTA 命名空间和文件时间归档；迁移表明确标记为
`phase_inferred_orphan_images`，不声称已确认具体实验。

## 路径改写与完整性

按用户要求，已批量把实验 JSON/JSONL、日志、配置、文档和关联脚本中的旧路径
改为 Phase 实体路径，并删除全部 217 个兼容软链接。图片引用也直接指向
Phase 目录。阶段明确的通配符和 f-string 路径一并处理。

改写文件中的路径会改变其字节哈希。因此脚本沿引用关系更新已确认的文件哈希，
并重新计算受影响的 Skill artifact envelope 校验值；Skill 内容、图像字节、
动作、分数、成本、数据划分和实验结论保持不变。
新哈希表示搬迁后的文件身份，不是新的预注册，也不证明重新运行了实验。
旧版预注册、源代码及文件身份仍以备份为准；未能与迁移前文件对应的历史
代码版本哈希不会被冒充为当前版本重新签署。原有版本不匹配仍可能阻止历史续跑。

脚本为 `scripts/migrate_experiment_paths.py`，默认只预览；首次执行迁移时加
`--apply`。已执行完毕，重复预览应报告零待改写文件：

```bash
python3 scripts/migrate_experiment_paths.py
```

路径改写记录位于 `running/migrations/phase_path_rewrite_20260908/`：

- `originals.tar.gz`：逐字节备份所有被改写文件的原版，包括原有工作区修改。
- `files.json`：改写文件清单及修改前后 SHA-256。
- `hash_mapping.json`：已确认的旧哈希与新哈希对应关系。
- `aliases.json`：已删除软链接的原路径及目标。
- `report.json`：备份校验、文件读回校验和链接删除结果。
- `independent_verification.json`：图片、结构化数据和 Phase5 磁盘复查结果。

备份和历史源码压缩包保留旧内容，不参与批量替换。迁移脚本自身及专门测试旧路径的
测试夹具也排除在改写范围外。文档中的历史缩写哈希仍可能表示备份原版的身份。

后续新实验应显式把输出目录设置为 `running/PhaseN/<experiment_id>/`。
当前 CLI 使用 Phase5 协议，其默认实验和冻结评测输出已改到
`running/Phase5/vista_skill/`。模拟器的新图片默认落盘仍由现有环境代码决定，
不代表此次已统一未来所有图片输出。

先前仅移动目录、保留软链接时的记录仍保存在 `running/migrations/phase_reorganization_20260908/`：

- `moves.json`：每项旧路径、新路径、阶段及归类依据。
- `files_before.jsonl`：迁移前每个文件的大小、时间、inode 和 SHA-256。
- `completed_moves.jsonl`：实际完成的移动。
- `verification.json`：第一次目录移动当时的完整性和旧路径兼容检查结果，
  不代表批量路径改写之后的文件哈希。

这些迁移清单和校验记录允许由 Git 跟踪；原版压缩备份保持本地，见下文 Git 规则。

## 通用工程测试

六个 `stock_v2_*` 无模型 API 工程验证目录已统一归入 `running/test/`，
内部路径和相关文档已更新，不保留软链接。迁移备份及校验位于
`running/test/_migration_20260908/`。

两个 `resume_v2_{hab,nav}_fixture/` 续跑测试也已归入 `running/test/`，
路径引用和接受结果的哈希已核对，最终仍为 2/2。
这次迁移的备份和清单在 `running/migrations/resume_fixture_relocation_20260908/`。
后续通用工程测试全部放在 `running/test/<test_id>/`；后续迁移记录统一放在
`running/migrations/<migration_id>/`，不在 `running/` 根目录新建散落的测试目录。

## Git 同步范围

`running/` 与 `EmbodiedBench/running/` 不再整体忽略。
两处的非图片实验记录（JSON/JSONL、日志、配置、汇总、Skill 和迁移清单）
允许被 Git 跟踪。图片扩展名和无扩展名的 `audit/input_images/` 缓存仍忽略；
压缩备份和运行锁文件也保持本地。备份内可能含有图片，不用压缩包绕过图片排除。
规则集中在仓库根 `.gitignore`，已移除 `EmbodiedBench/.gitignore` 中的
`*running` 整体排除。

图片路径和哈希仍保留在记录中；仅克隆 Git 仓库不会自动获得图片字节。
提交 API 日志前检查凭据，禁止提交密钥、下载数据集和模拟器资产。
本次只修改跟踪规则，不执行 `git add`、commit 或 push。

## 保持原位的输出

- `running/closed_feedback/`：其他会话正在使用，未移动、改写或建立替代链接。
- `running/embodiedbench_native_nav/` 和 `EmbodiedBench/running/eb_nav/`：
  原生 benchmark 基线及环境测试，不归为 VISTA 方法演化实验。
- `EmbodiedBench/running/eb_habitat/images/`：通用环境截图，无足够依据归入具体方法实验。

参考：[Phase1](experiment_log_phase1.md)、[Phase2](experiment_log_phase2.md)、
[Phase3](experiment_log_phase3.md)、[Phase4](experiment_log_phase4.md)、
[Phase5](experiment_log_phase5.md)。
