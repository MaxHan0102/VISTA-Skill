# 通用工程测试输出

这些运行使用固定动作，不调用真实模型 API，不属于模型性能实验。

| 目录 | 用途 |
|---|---|
| [stock_v2_hab_full_validation](stock_v2_hab_full_validation/) | Habitat：保留官方反馈的流程验证 |
| [stock_v2_hab_rgb_only_validation](stock_v2_hab_rgb_only_validation/) | Habitat：RGB-only 流程验证 |
| [stock_v2_hab_smoke](stock_v2_hab_smoke/) | Habitat：快速环境和日志检查 |
| [stock_v2_nav_full_validation](stock_v2_nav_full_validation/) | Navigation：保留官方反馈的流程验证 |
| [stock_v2_nav_rgb_only_validation](stock_v2_nav_rgb_only_validation/) | Navigation：RGB-only 流程验证 |
| [stock_v2_nav_smoke](stock_v2_nav_smoke/) | Navigation：快速环境和日志检查 |

原目录已移入本目录，相关路径同步更新，未保留软链接。迁移备份和校验见 `_migration_20260908/`。

续跑测试：

- [resume_v2_hab_fixture](resume_v2_hab_fixture/)：Habitat 中断后仅补跑缺失任务。
- [resume_v2_nav_fixture](resume_v2_nav_fixture/)：Navigation 中断后仅补跑缺失任务。

均为无模型 API 的固定策略测试，最终恢复为 2/2。迁移备份与校验见
`../migrations/resume_fixture_relocation_20260908/`。
