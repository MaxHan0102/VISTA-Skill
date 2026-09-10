# 实验输出索引

VISTA-Skill 方法实验按阶段归档。

| 阶段 | 主要文件／目录 | 原始文件数 | 原始数据大小 |
|---|---:|---:|---:|
| [Phase1](Phase1/) | 21 | 27247 | 3.496 GB |
| [Phase2](Phase2/) | 36 | 12127 | 1.461 GB |
| [Phase3](Phase3/) | 3 | 2936 | 0.395 GB |
| [Phase4](Phase4/) | 1 | 3165 | 0.579 GB |
| [Phase5](Phase5/) | 17 | 3039 | 0.342 GB |

旧路径引用已批量改成 Phase 实体路径，217 个兼容软链接已删除。原版与哈希映射保存在 `migrations/phase_path_rewrite_20260908/`。

[test/](test/) 收纳六个 `stock_v2_*` 工程测试及两个 `resume_v2_*_fixture` 续跑测试。

`closed_feedback/` 和原生 benchmark 基线保持原位。
完整说明见 `../docs/experiment_output_layout.md`；最新迁移校验见 `migrations/phase_path_rewrite_20260908/independent_verification.json`。

Git 允许跟踪两个输出根目录中的非图片实验记录；图片、无扩展名图片缓存、压缩备份和运行锁保持忽略。
